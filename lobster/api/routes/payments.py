"""
Lobster AI - Payment Routes
Stripe integration for credit purchases and subscription management.
"""

import os
import json
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

import stripe
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Create router instance
router = APIRouter()

# Configure Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY")

# Verify Stripe configuration
if not stripe.api_key:
    logger.warning("STRIPE_SECRET_KEY not found in environment variables")
    
if not STRIPE_PUBLISHABLE_KEY:
    logger.warning("STRIPE_PUBLISHABLE_KEY not found in environment variables")

# Enums
class PaymentType(str, Enum):
    CREDITS = "credits"
    SUBSCRIPTION = "subscription"
    ENTERPRISE = "enterprise"

# Credit packages configuration
CREDIT_PACKAGES = {
    50: {"price": 2500, "bonus": 0},      # $25.00
    100: {"price": 4500, "bonus": 5},     # $45.00 + 5 bonus
    250: {"price": 10000, "bonus": 25},   # $100.00 + 25 bonus
    500: {"price": 18000, "bonus": 70}    # $180.00 + 70 bonus
}

# Pydantic Models
class CreateCheckoutSessionRequest(BaseModel):
    paymentType: PaymentType
    planType: Optional[str] = None
    creditAmount: Optional[int] = None
    creditPrice: Optional[float] = None
    successUrl: str
    cancelUrl: str

class VerifySessionRequest(BaseModel):
    sessionId: str

class CheckoutSessionResponse(BaseModel):
    id: str
    url: str
    paymentType: str
    planType: Optional[str] = None
    creditAmount: Optional[int] = None

class VerifySessionResponse(BaseModel):
    verified: bool
    customerId: Optional[str] = None
    planType: Optional[str] = None
    creditAmount: Optional[int] = None
    status: str
    startDate: Optional[str] = None
    nextBillingDate: Optional[str] = None

class UserCreditsResponse(BaseModel):
    customer_id: str
    credits: int
    last_updated: str

# In-memory storage for MVP (replace with database in production)
user_sessions = {}
user_credits = {}

# Helper functions
def verify_webhook_signature(payload: bytes, sig_header: str) -> bool:
    """Verify Stripe webhook signature"""
    if not STRIPE_WEBHOOK_SECRET:
        logger.warning("No webhook secret configured")
        return True  # Allow for development
    
    try:
        stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        return True
    except ValueError:
        logger.error("Invalid payload")
        return False
    except stripe.error.SignatureVerificationError:
        logger.error("Invalid signature")
        return False

def create_stripe_price_for_credits(amount: int) -> str:
    """Create a Stripe price for credit package"""
    try:
        if amount not in CREDIT_PACKAGES:
            raise ValueError(f"Invalid credit amount: {amount}")
        
        package = CREDIT_PACKAGES[amount]
        
        # First create a product
        product = stripe.Product.create(
            name=f'{amount} Lobster AI Credits',
            description=f'{amount} credits' + (f' + {package["bonus"]} bonus credits' if package["bonus"] > 0 else '')
        )
        
        # Then create price for the product
        price = stripe.Price.create(
            unit_amount=package["price"],
            currency='usd',
            product=product.id,
        )
        return price.id
    except Exception as e:
        logger.error(f"Error creating Stripe price: {e}")
        raise HTTPException(status_code=500, detail="Failed to create price")

# Routes
@router.post("/checkout/create-session", response_model=CheckoutSessionResponse)
async def create_checkout_session(request: CreateCheckoutSessionRequest):
    """Create Stripe checkout session for payments"""
    try:
        if not stripe.api_key:
            raise HTTPException(status_code=500, detail="Stripe not configured")
        
        logger.info(f"Creating checkout session for {request.paymentType}")
        
        if request.paymentType == PaymentType.ENTERPRISE:
            raise HTTPException(status_code=400, detail="Enterprise plans require custom pricing")
        
        elif request.paymentType == PaymentType.CREDITS:
            # Handle credit purchase
            credit_amount = request.creditAmount if request.creditAmount and request.creditAmount in CREDIT_PACKAGES else 100
            
            # Create price for credits
            price_id = create_stripe_price_for_credits(credit_amount)
            
            # Create checkout session
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price': price_id,
                    'quantity': 1,
                }],
                mode='payment',
                success_url=request.successUrl,
                cancel_url=request.cancelUrl,
                metadata={
                    'payment_type': request.paymentType,
                    'credit_amount': credit_amount,
                    'bonus_credits': CREDIT_PACKAGES[credit_amount]["bonus"]
                }
            )
            
            # Store session info for verification
            user_sessions[session.id] = {
                'payment_type': request.paymentType,
                'credit_amount': credit_amount,
                'bonus_credits': CREDIT_PACKAGES[credit_amount]["bonus"],
                'created_at': datetime.utcnow().isoformat()
            }
            
            return CheckoutSessionResponse(
                id=session.id,
                url=session.url,
                paymentType=request.paymentType,
                creditAmount=credit_amount
            )
        
        elif request.paymentType == PaymentType.SUBSCRIPTION:
            raise HTTPException(status_code=501, detail="Subscription payments not yet implemented")
        
        else:
            raise HTTPException(status_code=400, detail="Invalid payment type")
            
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
    except Exception as e:
        logger.error(f"Error creating checkout session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create checkout session")

@router.post("/checkout/verify-session", response_model=VerifySessionResponse)
async def verify_session(request: VerifySessionRequest):
    """Verify Stripe checkout session"""
    try:
        if not stripe.api_key:
            raise HTTPException(status_code=500, detail="Stripe not configured")
        
        # Retrieve session from Stripe
        session = stripe.checkout.Session.retrieve(request.sessionId)
        
        if session.payment_status != 'paid':
            return VerifySessionResponse(
                verified=False,
                status='unpaid'
            )
        
        # Get stored session info
        session_info = user_sessions.get(request.sessionId, {})
        
        # For credit purchases, add credits to user account
        if session_info.get('payment_type') == PaymentType.CREDITS:
            customer_id = session.customer
            credit_amount = session_info.get('credit_amount', 0)
            bonus_credits = session_info.get('bonus_credits', 0)
            total_credits = credit_amount + bonus_credits
            
            # Add credits to user account (in-memory for MVP)
            if customer_id not in user_credits:
                user_credits[customer_id] = 0
            user_credits[customer_id] += total_credits
            
            logger.info(f"Added {total_credits} credits to customer {customer_id}")
            
            return VerifySessionResponse(
                verified=True,
                customerId=customer_id,
                planType='usage',
                creditAmount=total_credits,
                status='active',
                startDate=datetime.utcnow().isoformat()
            )
        
        return VerifySessionResponse(
            verified=True,
            customerId=session.customer,
            status='active',
            startDate=datetime.utcnow().isoformat()
        )
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error during verification: {e}")
        raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
    except Exception as e:
        logger.error(f"Error verifying session: {e}")
        raise HTTPException(status_code=500, detail="Failed to verify session")

@router.post("/webhooks/stripe")
async def stripe_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle Stripe webhooks"""
    try:
        payload = await request.body()
        sig_header = request.headers.get('stripe-signature')
        
        if not verify_webhook_signature(payload, sig_header):
            raise HTTPException(status_code=400, detail="Invalid signature")
        
        event = json.loads(payload)
        
        # Handle different event types
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            logger.info(f"Payment succeeded for session: {session['id']}")
            background_tasks.add_task(process_successful_payment, session)
            
        elif event['type'] == 'invoice.payment_failed':
            invoice = event['data']['object']
            logger.warning(f"Payment failed for invoice: {invoice['id']}")
            background_tasks.add_task(handle_payment_failure, invoice)
        
        return JSONResponse(content={"status": "success"})
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=400, detail="Webhook processing failed")

@router.get("/user/credits/{customer_id}", response_model=UserCreditsResponse)
async def get_user_credits(customer_id: str):
    """Get user credit balance"""
    credits = user_credits.get(customer_id, 0)
    return UserCreditsResponse(
        customer_id=customer_id,
        credits=credits,
        last_updated=datetime.utcnow().isoformat()
    )

async def process_successful_payment(session: Dict[str, Any]):
    """Process successful payment in background"""
    try:
        session_id = session['id']
        customer_id = session.get('customer')
        metadata = session.get('metadata', {})
        payment_type = metadata.get('payment_type')
        
        if payment_type == 'credits':
            credit_amount = int(metadata.get('credit_amount', 0))
            bonus_credits = int(metadata.get('bonus_credits', 0))
            total_credits = credit_amount + bonus_credits
            
            if customer_id not in user_credits:
                user_credits[customer_id] = 0
            user_credits[customer_id] += total_credits
            
            logger.info(f"Webhook: Added {total_credits} credits to customer {customer_id}")
        
    except Exception as e:
        logger.error(f"Error processing successful payment: {e}")

async def handle_payment_failure(invoice: Dict[str, Any]):
    """Handle payment failure in background"""
    try:
        customer_id = invoice.get('customer')
        logger.warning(f"Processing payment failure for customer: {customer_id}")
        
    except Exception as e:
        logger.error(f"Error handling payment failure: {e}")
