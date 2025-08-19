# 🦞 Lobster AI - Web Interface Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Lobster AI bioinformatics system as a web-accessible API service on AWS and integrating it with a TypeScript/React frontend. The system transforms the existing CLI application into a scalable web service suitable for small team testing environments.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  React Frontend │────│ Application LB  │────│   ECS Fargate   │
│  (TypeScript)   │    │   (HTTPS/WSS)   │    │  Lobster API    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                               ┌─────────────────┐
                                               │       EFS       │
                                               │  (Workspaces)   │
                                               └─────────────────┘
```

## 📋 Prerequisites

### Local Development
- Python 3.11+
- Docker and Docker Compose
- AWS CLI configured with appropriate permissions
- Node.js 18+ (for React frontend development)

### AWS Requirements
- AWS Account with billing enabled
- IAM permissions for ECS, VPC, ELB, EFS, CloudWatch, and S3
- Domain name (optional, for custom HTTPS)

## 🚀 Part 1: Launch Project on AWS

### Step 1: Prepare the Environment

1. **Clone and Setup Local Environment**
```bash
git clone https://github.com/your-org/lobster.git
cd lobster

# Create production environment file
cp .env .env.production

# Edit environment variables
nano .env.production
```

2. **Configure Production Environment Variables**
```bash
# API Configuration
LOBSTER_ENV=production
LOBSTER_API_HOST=0.0.0.0
LOBSTER_API_PORT=8000
LOBSTER_SESSION_TIMEOUT=3600
LOBSTER_MAX_SESSIONS=100

# LLM Provider Keys (Required)
OPENAI_API_KEY=your-openai-key
AWS_BEDROCK_ACCESS_KEY=your-aws-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret
NCBI_API_KEY=your-ncbi-key

# AWS Infrastructure
AWS_REGION=us-east-1
EFS_FILE_SYSTEM_ID=fs-xxxxxxxxx  # Will be created
S3_BUCKET_NAME=lobster-uploads-your-suffix

# Database (Optional - defaults to in-memory)
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-very-long-random-secret-key
CORS_ORIGINS=https://yourdomain.com,http://localhost:3000
```

### Step 2: Build and Test Locally

1. **Build the API Container**
```bash
# Build the API Docker image
docker build -f docker/Dockerfile.api -t lobster-api:latest .

# Test locally with Docker Compose
docker-compose -f docker/docker-compose.api.yml up
```

2. **Verify API Functionality**
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test session creation
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test-user"}'
```

### Step 3: Deploy AWS Infrastructure

1. **Create S3 Bucket for Uploads**
```bash
# Create S3 bucket for file uploads
aws s3 mb s3://lobster-uploads-your-suffix --region us-east-1

# Configure bucket policy for uploads
aws s3api put-bucket-policy --bucket lobster-uploads-your-suffix --policy file://aws/s3-bucket-policy.json
```

2. **Deploy CloudFormation Stack**
```bash
# Deploy the infrastructure stack
aws cloudformation deploy \
  --template-file aws/cloudformation-stack.yml \
  --stack-name lobster-infrastructure \
  --parameter-overrides \
    VPCCidr=10.0.0.0/16 \
    PublicSubnetCidr1=10.0.1.0/24 \
    PublicSubnetCidr2=10.0.2.0/24 \
    PrivateSubnetCidr1=10.0.3.0/24 \
    PrivateSubnetCidr2=10.0.4.0/24 \
  --capabilities CAPABILITY_IAM
```

3. **Push Container to ECR**
```bash
# Get ECR login token
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

# Create ECR repository
aws ecr create-repository --repository-name lobster-api --region us-east-1

# Tag and push image
docker tag lobster-api:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/lobster-api:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/lobster-api:latest
```

### Step 4: Deploy ECS Service

1. **Create ECS Cluster and Service**
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name lobster-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://aws/ecs-task-definition.json

# Create ECS service
aws ecs create-service \
  --cluster lobster-cluster \
  --service-name lobster-api-service \
  --task-definition lobster-api:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxx,subnet-yyyyy],securityGroups=[sg-zzzzz],assignPublicIp=ENABLED}"
```

2. **Configure Application Load Balancer**
```bash
# Get ALB DNS name from CloudFormation outputs
aws cloudformation describe-stacks \
  --stack-name lobster-infrastructure \
  --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDNS`].OutputValue' \
  --output text
```

### Step 5: Configure HTTPS (Optional)

1. **Request SSL Certificate**
```bash
# Request ACM certificate
aws acm request-certificate \
  --domain-name api.yourdomain.com \
  --subject-alternative-names yourdomain.com \
  --validation-method DNS \
  --region us-east-1
```

2. **Update ALB Listener for HTTPS**
```bash
# Add HTTPS listener to ALB (update listener ARN)
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:elasticloadbalancing:us-east-1:xxx:loadbalancer/app/xxx \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=arn:aws:acm:us-east-1:xxx:certificate/xxx \
  --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:us-east-1:xxx:targetgroup/xxx
```

### Step 6: Verify Deployment

1. **Test API Endpoints**
```bash
# Replace with your ALB DNS name
API_BASE_URL="https://your-alb-dns-name.us-east-1.elb.amazonaws.com"

# Test health endpoint
curl "$API_BASE_URL/health"

# Test session creation
curl -X POST "$API_BASE_URL/api/v1/sessions" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "production-test"}'
```

2. **Monitor Logs**
```bash
# View CloudWatch logs
aws logs tail /ecs/lobster-api --follow
```

## 🎨 Part 2: TypeScript/React Frontend Integration

### Step 1: Create React Application

1. **Initialize React Project with TypeScript**
```bash
# Create new React app with TypeScript
npx create-react-app lobster-frontend --template typescript
cd lobster-frontend

# Install additional dependencies
npm install \
  @tanstack/react-query \
  axios \
  socket.io-client \
  @mui/material \
  @emotion/react \
  @emotion/styled \
  @mui/icons-material \
  react-router-dom \
  @types/node
```

2. **Install Bioinformatics-Specific Packages**
```bash
# For data visualization
npm install plotly.js react-plotly.js @types/plotly.js

# For file uploads and data handling
npm install react-dropzone papaparse @types/papaparse

# For code syntax highlighting (for exported analyses)
npm install react-syntax-highlighter @types/react-syntax-highlighter
```

### Step 2: Configure API Client

1. **Create API Configuration (`src/config/api.ts`)**
```typescript
export const API_CONFIG = {
  BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  WS_URL: process.env.REACT_APP_WS_URL || 'ws://localhost:8000',
  ENDPOINTS: {
    SESSIONS: '/api/v1/sessions',
    CHAT: '/api/v1/chat',
    DATA: '/api/v1/data',
    PLOTS: '/api/v1/plots',
    FILES: '/api/v1/files',
    HEALTH: '/health'
  }
} as const;
```

2. **Create API Client (`src/services/apiClient.ts`)**
```typescript
import axios, { AxiosInstance } from 'axios';
import { API_CONFIG } from '../config/api';

export interface SessionResponse {
  session_id: string;
  created_at: string;
  last_active: string;
  user_id?: string;
  workspace_path: string;
  status: string;
}

export interface ChatRequest {
  message: string;
  session_id: string;
  stream?: boolean;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  message_id: string;
  plots?: PlotInfo[];
  data_updated: boolean;
}

export interface PlotInfo {
  id: string;
  title: string;
  timestamp: string;
  source: string;
  format: string;
}

class LobsterAPIClient {
  private client: AxiosInstance;
  private currentSessionId: string | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: API_CONFIG.BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for adding session ID
    this.client.interceptors.request.use((config) => {
      if (this.currentSessionId && config.headers) {
        config.headers['X-Session-ID'] = this.currentSessionId;
      }
      return config;
    });
  }

  // Session Management
  async createSession(userId?: string): Promise<SessionResponse> {
    const response = await this.client.post<SessionResponse>(
      API_CONFIG.ENDPOINTS.SESSIONS,
      { user_id: userId }
    );
    this.currentSessionId = response.data.session_id;
    return response.data;
  }

  async getSession(sessionId: string): Promise<SessionResponse> {
    const response = await this.client.get<SessionResponse>(
      `${API_CONFIG.ENDPOINTS.SESSIONS}/${sessionId}`
    );
    return response.data;
  }

  // Chat Interface
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    const response = await this.client.post<ChatResponse>(
      API_CONFIG.ENDPOINTS.CHAT,
      request
    );
    return response.data;
  }

  async getChatHistory(sessionId: string, limit = 50) {
    const response = await this.client.get(
      `${API_CONFIG.ENDPOINTS.CHAT}/${sessionId}/history`,
      { params: { limit } }
    );
    return response.data;
  }

  // File Upload
  async uploadFile(file: File, sessionId: string) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);

    const response = await this.client.post(
      API_CONFIG.ENDPOINTS.FILES,
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
      }
    );
    return response.data;
  }

  // Data Management
  async downloadGEODataset(geoId: string, sessionId: string) {
    const response = await this.client.post(
      `${API_CONFIG.ENDPOINTS.DATA}/geo`,
      { geo_id: geoId, session_id: sessionId }
    );
    return response.data;
  }

  async getDatasetInfo(sessionId: string) {
    const response = await this.client.get(
      `${API_CONFIG.ENDPOINTS.DATA}/${sessionId}`
    );
    return response.data;
  }

  // Plots
  async getPlots(sessionId: string) {
    const response = await this.client.get(
      `${API_CONFIG.ENDPOINTS.PLOTS}/${sessionId}`
    );
    return response.data;
  }

  async getPlotData(plotId: string, sessionId: string) {
    const response = await this.client.get(
      `${API_CONFIG.ENDPOINTS.PLOTS}/${sessionId}/${plotId}/data`
    );
    return response.data;
  }

  // Health Check
  async healthCheck() {
    const response = await this.client.get(API_CONFIG.ENDPOINTS.HEALTH);
    return response.data;
  }

  // Getters
  get sessionId() {
    return this.currentSessionId;
  }
}

export const apiClient = new LobsterAPIClient();
```

### Step 3: Create WebSocket Connection

1. **WebSocket Service (`src/services/websocketService.ts`)**
```typescript
import { io, Socket } from 'socket.io-client';
import { API_CONFIG } from '../config/api';

export enum WSEventType {
  CHAT_STREAM = 'chat_stream',
  AGENT_THINKING = 'agent_thinking',
  ANALYSIS_PROGRESS = 'analysis_progress',
  DATA_UPDATED = 'data_updated',
  PLOT_GENERATED = 'plot_generated',
  ERROR = 'error'
}

export interface WSMessage {
  event_type: WSEventType;
  session_id: string;
  data: any;
  timestamp: string;
}

class WebSocketService {
  private socket: Socket | null = null;
  private sessionId: string | null = null;

  connect(sessionId: string) {
    if (this.socket?.connected) {
      this.disconnect();
    }

    this.sessionId = sessionId;
    this.socket = io(API_CONFIG.WS_URL, {
      query: { session_id: sessionId },
      transports: ['websocket'],
    });

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
    });

    this.socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
    });

    return this.socket;
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  onMessage(eventType: WSEventType, callback: (data: any) => void) {
    this.socket?.on(eventType, callback);
  }

  offMessage(eventType: WSEventType, callback?: (data: any) => void) {
    this.socket?.off(eventType, callback);
  }

  sendMessage(eventType: WSEventType, data: any) {
    if (this.socket && this.sessionId) {
      this.socket.emit(eventType, {
        event_type: eventType,
        session_id: this.sessionId,
        data,
        timestamp: new Date().toISOString(),
      });
    }
  }

  get isConnected() {
    return this.socket?.connected || false;
  }
}

export const wsService = new WebSocketService();
```

### Step 4: Build Core React Components

1. **Main Application (`src/App.tsx`)**
```tsx
import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

import { SessionProvider } from './contexts/SessionContext';
import { ChatInterface } from './components/ChatInterface';
import { DataUpload } from './components/DataUpload';
import { PlotViewer } from './components/PlotViewer';
import { Sidebar } from './components/Sidebar';
import { apiClient } from './services/apiClient';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#d32f2f', // Lobster red
    },
    secondary: {
      main: '#424242',
    },
  },
});

const queryClient = new QueryClient();

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Initialize session on app start
    const initializeSession = async () => {
      try {
        const session = await apiClient.createSession();
        setSessionId(session.session_id);
      } catch (error) {
        console.error('Failed to create session:', error);
      } finally {
        setLoading(false);
      }
    };

    initializeSession();
  }, []);

  if (loading) {
    return <div>Loading Lobster AI...</div>;
  }

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <SessionProvider sessionId={sessionId}>
          <Router>
            <div style={{ display: 'flex', height: '100vh' }}>
              <Sidebar />
              <main style={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                <Routes>
                  <Route path="/" element={<ChatInterface />} />
                  <Route path="/upload" element={<DataUpload />} />
                  <Route path="/plots" element={<PlotViewer />} />
                </Routes>
              </main>
            </div>
          </Router>
        </SessionProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
```

2. **Chat Interface (`src/components/ChatInterface.tsx`)**
```tsx
import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  CircularProgress,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import { useSession } from '../contexts/SessionContext';
import { apiClient } from '../services/apiClient';
import { wsService, WSEventType } from '../services/websocketService';

interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  plots?: any[];
}

export const ChatInterface: React.FC = () => {
  const { sessionId } = useSession();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (sessionId) {
      // Connect WebSocket
      wsService.connect(sessionId);

      // Listen for streaming responses
      wsService.onMessage(WSEventType.CHAT_STREAM, (data) => {
        setMessages((prev) => {
          const lastMessage = prev[prev.length - 1];
          if (lastMessage && lastMessage.role === 'assistant') {
            return [
              ...prev.slice(0, -1),
              { ...lastMessage, content: lastMessage.content + data.content }
            ];
          } else {
            return [
              ...prev,
              {
                role: 'assistant',
                content: data.content,
                timestamp: new Date().toISOString()
              }
            ];
          }
        });
      });

      // Listen for agent thinking
      wsService.onMessage(WSEventType.AGENT_THINKING, (data) => {
        console.log('Agent thinking:', data);
      });

      return () => {
        wsService.disconnect();
      };
    }
  }, [sessionId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!currentMessage.trim() || !sessionId) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: currentMessage,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setCurrentMessage('');
    setLoading(true);

    try {
      const response = await apiClient.sendMessage({
        message: currentMessage,
        session_id: sessionId,
        stream: true,
      });

      if (!response) {
        // Response will come through WebSocket
        return;
      }

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.response,
        timestamp: new Date().toISOString(),
        plots: response.plots,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: ChatMessage = {
        role: 'system',
        content: 'Error: Unable to send message. Please try again.',
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', p: 2 }}>
      <Typography variant="h4" gutterBottom>
        🦞 Lobster AI Assistant
      </Typography>
      
      <Paper
        sx={{
          flexGrow: 1,
          overflow: 'auto',
          p: 2,
          mb: 2,
          backgroundColor: '#f5f5f5',
        }}
      >
        {messages.map((message, index) => (
          <Box
            key={index}
            sx={{
              mb: 2,
              display: 'flex',
              justifyContent:
                message.role === 'user' ? 'flex-end' : 'flex-start',
            }}
          >
            <Paper
              sx={{
                p: 2,
                maxWidth: '70%',
                backgroundColor:
                  message.role === 'user'
                    ? 'primary.main'
                    : message.role === 'system'
                    ? 'error.main'
                    : 'background.paper',
                color:
                  message.role === 'user' || message.role === 'system'
                    ? 'white'
                    : 'text.primary',
              }}
            >
              <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                {message.content}
              </Typography>
              {message.plots && message.plots.length > 0 && (
                <Typography variant="caption" color="text.secondary">
                  Generated {message.plots.length} plot(s)
                </Typography>
              )}
            </Paper>
          </Box>
        ))}
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
            <CircularProgress />
          </Box>
        )}
        <div ref={messagesEndRef} />
      </Paper>

      <Box sx={{ display: 'flex', gap: 1 }}>
        <TextField
          fullWidth
          multiline
          maxRows={4}
          value={currentMessage}
          onChange={(e) => setCurrentMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about bioinformatics analysis, upload data, or request help..."
          disabled={loading}
        />
        <Button
          variant="contained"
          onClick={sendMessage}
          disabled={loading || !currentMessage.trim()}
          endIcon={<SendIcon />}
        >
          Send
        </Button>
      </Box>
    </Box>
  );
};
```

### Step 5: Environment Configuration

1. **Create Environment Files**
```bash
# Development environment (.env.development)
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000

# Production environment (.env.production)
REACT_APP_API_URL=https://your-alb-dns-name.us-east-1.elb.amazonaws.com
REACT_APP_WS_URL=wss://your-alb-dns-name.us-east-1.elb.amazonaws.com
```

2. **Build and Deploy Frontend**
```bash
# Build for production
npm run build

# Deploy to S3 (example)
aws s3 sync build/ s3://your-frontend-bucket --delete

# Or deploy to Netlify/Vercel
# netlify deploy --prod --dir=build
```

## 🔧 Configuration Examples

### Environment Variables Reference
```bash
# API Configuration
LOBSTER_ENV=production
LOBSTER_API_HOST=0.0.0.0
LOBSTER_API_PORT=8000
LOBSTER_SESSION_TIMEOUT=3600
LOBSTER_MAX_SESSIONS=100
LOBSTER_WORKSPACE_ROOT=/app/workspaces

# Security
SECRET_KEY=your-very-long-random-secret-key-min-32-chars
CORS_ORIGINS=https://yourdomain.com,http://localhost:3000

# AWS Services
AWS_REGION=us-east-1
S3_BUCKET_NAME=lobster-uploads-your-suffix
EFS_FILE_SYSTEM_ID=fs-xxxxxxxxx

# LLM Providers (Required)
OPENAI_API_KEY=sk-...
AWS_BEDROCK_ACCESS_KEY=AKIA...
AWS_BEDROCK_SECRET_ACCESS_KEY=...
NCBI_API_KEY=...

# Optional: Redis for session storage
REDIS_URL=redis://redis-cluster.xxx.cache.amazonaws.com:6379

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO
```

### Sample API Usage
```typescript
// Initialize session
const session = await apiClient.createSession('user123');

// Upload data file
const file = document.getElementById('fileInput').files[0];
await apiClient.uploadFile(file, session.session_id);

// Send analysis request
const response = await apiClient.sendMessage({
  message: 'Run quality control on the uploaded dataset',
  session_id: session.session_id
});

// Download GEO dataset
await apiClient.downloadGEODataset('GSE109564', session.session_id);
```

## 🚨 Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Verify security groups allow traffic on port 8000
   - Check ALB target group health
   - Validate environment variables in ECS task

2. **WebSocket Connection Failed**
   - Ensure ALB supports WebSocket upgrades
   - Check CORS configuration
   - Verify WebSocket URL protocol (ws/wss)

3. **File Upload Issues**
   - Check S3 bucket permissions
   - Verify EFS mount points
   - Increase upload size limits in ALB

4. **Session Management Problems**
   - Monitor ECS task memory usage
   - Check session timeout configuration
   - Verify Redis connection if using external storage

### Monitoring and Logs
```bash
# View ECS service logs
aws logs tail /ecs/lobster-api --follow

# Check ECS service health
aws ecs describe-services --cluster lobster-cluster --services lobster-api-service

# Monitor ALB target health
aws elbv2 describe-target-health --target-group-arn your-target-group-arn
```

## 📈 Scaling Considerations

### For Small Teams (2-10 users)
- **ECS Tasks**: 2-4 tasks
- **Instance Size**: 1 vCPU, 2GB RAM per task
- **Storage**: 20GB EFS
- **Database**: In-memory sessions (sufficient)

### For Medium Usage (10-50 users)
- **ECS Tasks**: 4-8 tasks with auto-scaling
- **Instance Size**: 2 vCPU, 4GB RAM per task
- **Storage**: 100GB EFS with provisioned throughput
- **Database**: Redis cluster for session storage

### Cost Optimization
- Use Spot instances for development environments
- Schedule ECS services to scale down during off-hours
- Implement session cleanup to reduce storage costs
- Use S3 lifecycle policies for old uploads

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [AWS ECS Fargate Guide](https://docs.aws.amazon.com/ecs/latest/userguide/AWS_Fargate.html)
- [React TypeScript Handbook](https://react-typescript-cheatsheet.netlify.app/)
- [WebSocket Implementation Guide](https://socket.io/docs/v4/)

---

This deployment guide provides a production-ready setup for the Lobster AI bioinformatics system. The architecture is designed for reliability, scalability, and ease of maintenance while keeping costs reasonable for small team environments.
