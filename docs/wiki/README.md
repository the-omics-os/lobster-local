# Lobster AI Documentation Wiki

This directory contains the complete documentation for Lobster AI, designed to be published as a GitHub Wiki.

## 📂 Documentation Structure

The documentation is organized into 30 comprehensive markdown files covering:

- **Getting Started** (Files 01-03): Installation, configuration, quick start
- **User Guide** (Files 04-07): How to use Lobster AI for analysis
- **Developer Guide** (Files 08-12): Extending and contributing to Lobster AI
- **API Reference** (Files 13-17): Complete API documentation
- **Architecture** (Files 18-22): System design and internals
- **Tutorials** (Files 23-27): Practical examples and workflows
- **Support** (Files 28-30): Troubleshooting, FAQ, glossary

## 🚀 Publishing to GitHub Wiki

### Option 1: Automatic Wiki Setup (Recommended)

1. Enable Wiki in your GitHub repository settings
2. Clone the wiki repository:
   ```bash
   git clone https://github.com/homara-ai/lobster.wiki.git
   cd lobster.wiki
   ```

3. Copy all documentation files:
   ```bash
   cp /path/to/lobster/docs/wiki/*.md .
   ```

4. Commit and push:
   ```bash
   git add .
   git commit -m "Add comprehensive Lobster AI documentation"
   git push origin master
   ```

### Option 2: Manual Upload via GitHub Web Interface

1. Go to your repository's Wiki tab
2. Click "Create the first page" or "New Page"
3. Copy content from `Home.md` as the main page
4. Create additional pages for each documentation file
5. Use the file names (without `.md`) as page titles

### Option 3: Using GitHub Actions (CI/CD)

Create `.github/workflows/wiki-sync.yml`:

```yaml
name: Sync Wiki Documentation

on:
  push:
    paths:
      - 'docs/wiki/**'
    branches:
      - main

jobs:
  sync-wiki:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Git
        run: |
          git config --global user.name 'GitHub Actions'
          git config --global user.email 'actions@github.com'

      - name: Sync to Wiki
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          git clone https://$GITHUB_TOKEN@github.com/${{ github.repository }}.wiki.git wiki
          cp docs/wiki/*.md wiki/
          cd wiki
          git add .
          git diff --staged --quiet || git commit -m "Sync documentation from main repository"
          git push
```

## 📝 Documentation Files

| File | Title | Description |
|------|-------|-------------|
| `Home.md` | Main Wiki Page | Entry point with navigation |
| `01-getting-started.md` | Getting Started | 5-minute quick start guide |
| `02-installation.md` | Installation | Comprehensive setup instructions |
| `03-configuration.md` | Configuration | Environment and API setup |
| `04-user-guide-overview.md` | User Guide Overview | How Lobster AI works |
| `05-cli-commands.md` | CLI Commands | Complete command reference |
| `06-data-analysis-workflows.md` | Analysis Workflows | Step-by-step guides |
| `07-data-formats.md` | Data Formats | Supported formats |
| `08-developer-overview.md` | Developer Overview | Architecture and setup |
| `09-creating-agents.md` | Creating Agents | Agent development guide |
| `10-creating-services.md` | Creating Services | Service implementation |
| `11-creating-adapters.md` | Creating Adapters | Data adapter guide |
| `12-testing-guide.md` | Testing Guide | Test framework |
| `13-api-overview.md` | API Overview | API introduction |
| `14-core-api.md` | Core API | DataManagerV2 and clients |
| `15-agents-api.md` | Agents API | Agent tools reference |
| `16-services-api.md` | Services API | Service interfaces |
| `17-interfaces-api.md` | Interfaces API | Abstract interfaces |
| `18-architecture-overview.md` | Architecture Overview | System design |
| `19-agent-system.md` | Agent System | Multi-agent architecture |
| `20-data-management.md` | Data Management | DataManagerV2 design |
| `21-cloud-local-architecture.md` | Cloud/Local | Deployment architecture |
| `22-performance-optimization.md` | Performance | Optimization strategies |
| `23-tutorial-single-cell.md` | Single-Cell Tutorial | scRNA-seq workflow |
| `24-tutorial-bulk-rnaseq.md` | Bulk RNA-seq Tutorial | DE analysis guide |
| `25-tutorial-proteomics.md` | Proteomics Tutorial | MS/affinity workflows |
| `26-tutorial-custom-agent.md` | Custom Agent Tutorial | Agent creation example |
| `27-examples-cookbook.md` | Examples Cookbook | Code recipes |
| `28-troubleshooting.md` | Troubleshooting | Problem solutions |
| `29-faq.md` | FAQ | Common questions |
| `30-glossary.md` | Glossary | Term definitions |

## 🔧 Maintaining Documentation

### Updating Documentation

1. Edit the markdown files in `docs/wiki/`
2. Test locally with a markdown viewer
3. Commit changes to the main repository
4. Sync to wiki (manual or automated)

### Documentation Standards

- Use clear, concise language
- Include code examples with syntax highlighting
- Add diagrams using Mermaid when helpful
- Cross-reference related topics with links
- Keep examples up-to-date with the codebase

### Adding New Documentation

1. Create new markdown file with sequential numbering
2. Update `Home.md` with link to new page
3. Update this README with the new file entry
4. Follow the established formatting patterns

## 🎨 Formatting Guidelines

### Headers
- Use `#` for page title
- Use `##` for main sections
- Use `###` for subsections
- Use `####` for detailed points

### Code Blocks
```python
# Use syntax highlighting
def example():
    return "formatted code"
```

### Links
- Internal: `[Link Text](file-name.md)`
- Sections: `[Link Text](file-name.md#section-header)`
- External: `[Link Text](https://example.com)`

### Tables
| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |

### Alerts
**Note:** Important information
**Warning:** Caution required
**Tip:** Helpful suggestion

## 📊 Documentation Statistics

- **Total Files**: 31 (including Home.md)
- **Total Size**: ~500KB
- **Coverage**: Complete platform documentation
- **Examples**: 50+ code examples
- **Tutorials**: 5 comprehensive tutorials
- **API Methods**: 100+ documented

## 🔗 Additional Resources

- [Main Repository](https://github.com/homara-ai/lobster)
- [Issue Tracker](https://github.com/homara-ai/lobster/issues)
- [Discord Community](https://discord.gg/homaraai)
- [Enterprise Support](mailto:enterprise@homara.ai)

## 📄 License

This documentation is part of the Lobster AI project and is licensed under the MIT License.

---

*Documentation created for Lobster AI v2.2+*