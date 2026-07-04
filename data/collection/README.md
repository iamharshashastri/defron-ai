# Data Collection

This directory contains scripts for collecting and processing cybersecurity training data.

## Data Sources

- NVD (National Vulnerability Database)
- ArXiv Security Research Papers
- MITRE ATT&CK Framework
- Public Threat Intelligence Feeds
- Security Documentation and Resources

## Scripts

- `collect_cve_data.py` - Collect CVE records from public databases
- `collect_research_papers.py` - Scrape security research papers
- `preprocess.py` - Clean and normalize data
- `validate.py` - Validate data quality and licensing

## Usage

```bash
python collect_cve_data.py
python collect_research_papers.py
python preprocess.py
python validate.py
```

## Data Pipeline

1. **Collection** - Download from public sources
2. **Validation** - Check licensing and data integrity
3. **Cleaning** - Remove duplicates and normalize
4. **Tokenization** - Prepare for model training
