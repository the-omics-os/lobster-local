#!/usr/bin/env python3
"""
Basic Single-Cell RNA-seq Analysis with Lobster AI

This example demonstrates how to:
1. Download a dataset from GEO
2. Perform quality control and filtering
3. Run clustering analysis
4. Generate visualizations

Requirements:
- Lobster AI installed locally
- API keys configured in .env file
"""

import os
from pathlib import Path

# Set up environment
os.environ.setdefault("GENIE_PROFILE", "production")

def main():
    """Run basic single-cell analysis example."""
    
    print("🦞 Lobster AI - Basic Single-Cell RNA-seq Example")
    print("=" * 55)
    
    try:
        # Import Lobster AI components
        from lobster.core.client import AgentClient
        from lobster.core.data_manager_v2 import DataManagerV2
        
        # Initialize client
        print("🔧 Initializing Lobster AI...")
        workspace = Path("./example_workspace")
        data_manager = DataManagerV2(workspace_path=workspace)
        client = AgentClient(data_manager=data_manager)
        
        # Example queries to demonstrate capabilities
        examples = [
            {
                "title": "📥 Download Single-Cell Dataset",
                "query": "Download GSE109564 from GEO - it's a single-cell RNA-seq dataset of immune cells",
                "description": "This will download and load a real single-cell dataset"
            },
            {
                "title": "📊 Quality Assessment",
                "query": "Assess the quality of the loaded data and show me basic statistics",
                "description": "Check data quality metrics like genes per cell, mitochondrial content"
            },
            {
                "title": "🔬 Clustering Analysis", 
                "query": "Filter low-quality cells, normalize the data, and perform clustering analysis",
                "description": "Complete preprocessing and clustering pipeline"
            },
            {
                "title": "🎯 Marker Gene Analysis",
                "query": "Find marker genes for each cluster and create visualizations",
                "description": "Identify distinctive genes for each cell population"
            }
        ]
        
        print("\n🚀 Running analysis pipeline...\n")
        
        # Run each example query
        for i, example in enumerate(examples, 1):
            print(f"{'='*60}")
            print(f"{i}/4 {example['title']}")
            print(f"    {example['description']}")
            print(f"{'='*60}")
            
            # Process the query
            result = client.query(example['query'])
            
            if result['success']:
                print(f"✅ {example['title']} - Complete!")
                print(f"📝 Response preview: {result['response'][:200]}...")
                
                # Show any plots that were generated
                if result.get('plots'):
                    print(f"📊 Generated {len(result['plots'])} visualization(s)")
                
            else:
                print(f"❌ {example['title']} - Failed: {result['error']}")
                print("   Continuing with next step...")
            
            print()
        
        # Summary
        print("🎉 Example analysis complete!")
        print(f"📁 Results saved to: {workspace}")
        print("📊 Check the plots directory for visualizations")
        print()
        print("💡 Next steps:")
        print("   • Try 'lobster chat' for interactive analysis")
        print("   • Modify this script for your own data")
        print("   • Explore other examples in this directory")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure Lobster AI is installed:")
        print("   pip install -e .")
        return 1
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("💡 Check your .env file has valid API keys")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
