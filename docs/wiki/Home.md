# 🦞 Lobster AI Documentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Welcome to the comprehensive documentation for **Lobster AI** - the AI-powered multi-omics bioinformatics analysis platform. This documentation provides everything you need to use, develop, and extend Lobster AI.

## 📚 Documentation Structure

### 🚀 **Getting Started**
Start here if you're new to Lobster AI
- [**01 - Getting Started**](01-getting-started.md) - Quick 5-minute setup guide
- [**02 - Installation**](02-installation.md) - Comprehensive installation instructions
- [**03 - Configuration**](03-configuration.md) - API keys, environment setup, and model profiles

### 👤 **User Guide**
Learn how to use Lobster AI for your research
- [**04 - User Guide Overview**](04-user-guide-overview.md) - Understanding how Lobster AI works
- [**05 - CLI Commands**](05-cli-commands.md) - Complete command reference with examples
- [**06 - Data Analysis Workflows**](06-data-analysis-workflows.md) - Step-by-step analysis guides
- [**07 - Data Formats**](07-data-formats.md) - Supported input/output formats

### 💻 **Developer Guide**
Extend and contribute to Lobster AI
- [**08 - Developer Overview**](08-developer-overview.md) - Architecture and development setup
- [**09 - Creating Agents**](09-creating-agents.md) - Build new specialized AI agents
- [**10 - Creating Services**](10-creating-services.md) - Implement analysis services
- [**11 - Creating Adapters**](11-creating-adapters.md) - Add support for new data formats
- [**12 - Testing Guide**](12-testing-guide.md) - Writing and running tests

### 📖 **API Reference**
Complete API documentation
- [**13 - API Overview**](13-api-overview.md) - API organization and conventions
- [**14 - Core API**](14-core-api.md) - DataManagerV2 and client interfaces
- [**15 - Agents API**](15-agents-api.md) - Agent tools and capabilities
- [**16 - Services API**](16-services-api.md) - Analysis service interfaces
- [**17 - Interfaces API**](17-interfaces-api.md) - Abstract interfaces and contracts

### 🏗️ **Architecture & Internals**
Deep dive into system design
- [**18 - Architecture Overview**](18-architecture-overview.md) - System design and components
- [**19 - Agent System**](19-agent-system.md) - Multi-agent coordination architecture
- [**20 - Data Management**](20-data-management.md) - DataManagerV2 and modality system
- [**21 - Cloud/Local Architecture**](21-cloud-local-architecture.md) - Hybrid deployment design
- [**22 - Performance Optimization**](22-performance-optimization.md) - Memory and speed optimizations

### 🎯 **Tutorials & Examples**
Learn by doing with practical tutorials
- [**23 - Single-Cell RNA-seq Tutorial**](23-tutorial-single-cell.md) - Complete workflow with real data
- [**24 - Bulk RNA-seq Tutorial**](24-tutorial-bulk-rnaseq.md) - Differential expression analysis
- [**25 - Proteomics Tutorial**](25-tutorial-proteomics.md) - MS and affinity proteomics
- [**26 - Custom Agent Tutorial**](26-tutorial-custom-agent.md) - Create your own agent
- [**27 - Examples Cookbook**](27-examples-cookbook.md) - Code recipes and patterns

### 🔧 **Support & Reference**
Help and additional resources
- [**28 - Troubleshooting**](28-troubleshooting.md) - Common issues and solutions
- [**29 - FAQ**](29-faq.md) - Frequently asked questions
- [**30 - Glossary**](30-glossary.md) - Bioinformatics and technical terms

## 🎯 Quick Navigation by Task

### **"I want to..."**

#### **Get Started Quickly**
- [Install Lobster AI in 5 minutes](01-getting-started.md)
- [Configure my API keys](03-configuration.md#required-api-keys)
- [Run my first analysis](01-getting-started.md#your-first-analysis)

#### **Analyze My Data**
- [Analyze single-cell RNA-seq data](23-tutorial-single-cell.md)
- [Perform bulk RNA-seq differential expression](24-tutorial-bulk-rnaseq.md)
- [Process proteomics data](25-tutorial-proteomics.md)
- [Download and analyze GEO datasets](06-data-analysis-workflows.md#geo-database-integration)

#### **Understand the System**
- [Learn about the architecture](18-architecture-overview.md)
- [Understand how agents work](19-agent-system.md)
- [See supported data formats](07-data-formats.md)

#### **Extend Lobster AI**
- [Create a new agent](09-creating-agents.md)
- [Add a new analysis service](10-creating-services.md)
- [Support a new data format](11-creating-adapters.md)
- [Contribute to the project](08-developer-overview.md#contributing)

#### **Solve Problems**
- [Fix installation issues](28-troubleshooting.md#installation-issues)
- [Resolve data loading errors](28-troubleshooting.md#data-loading-issues)
- [Debug analysis failures](28-troubleshooting.md#analysis-failures)

## 🌟 Key Features

### **🤖 AI-Powered Analysis**
- Natural language interface for complex bioinformatics
- 8+ specialized AI agents for different analysis domains
- Intelligent workflow coordination and parameter optimization

### **🧬 Scientific Capabilities**
- **Single-Cell RNA-seq**: QC, clustering, annotation, trajectory analysis
- **Bulk RNA-seq**: pyDESeq2 differential expression with complex designs
- **Proteomics**: MS/affinity analysis with missing value handling
- **Multi-Omics**: Integrated cross-platform analysis

### **☁️ Deployment Flexibility**
- **Local Mode**: Full privacy with data on your machine
- **Cloud Mode**: Scalable computing with managed infrastructure
- **Hybrid**: Automatic switching between modes

### **📊 Professional Features**
- Publication-ready visualizations
- W3C-PROV compliant provenance tracking
- Comprehensive quality control metrics
- Batch effect detection and correction

## 📈 Version Highlights

### **v2.2+ Features**
- 🔄 **Workspace Restoration** - Seamless session continuity
- 📂 **Pattern-based Dataset Loading** - Smart memory management
- 💾 **Session Persistence** - Automatic state tracking

### **v2.1+ Features**
- ⌨️ **Enhanced CLI** - Arrow navigation and command history
- 🎨 **Rich Interface** - Professional orange branding
- ⚡ **Performance** - Optimized startup and processing

## 🔗 Quick Links

- **GitHub Repository**: [github.com/homara-ai/lobster](https://github.com/homara-ai/lobster)
- **Issue Tracker**: [Report bugs or request features](https://github.com/homara-ai/lobster/issues)
- **Discord Community**: [Join our community](https://discord.gg/homaraai)
- **Enterprise Support**: [enterprise@homara.ai](mailto:enterprise@homara.ai)

## 📝 Documentation Standards

This documentation follows these principles:
- **Progressive Disclosure**: Start simple, dive deeper as needed
- **Task-Oriented**: Organized by what you want to accomplish
- **Example-Rich**: Real datasets and practical code examples
- **Cross-Referenced**: Links between related topics
- **Maintained**: Regular updates with each release

## 🤝 Contributing to Documentation

Found an issue or want to improve the documentation?
1. Check our [contribution guidelines](08-developer-overview.md#contributing)
2. Submit a pull request to the `docs/wiki` directory
3. Follow our [documentation style guide](08-developer-overview.md#documentation-style)

---

*Documentation for Lobster AI v2.2+ | Last updated: 2025*

*Made with ❤️ by [Homara AI](https://homara.ai)*