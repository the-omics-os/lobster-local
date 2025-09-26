# Lobster AI Agents - Detailed Fix Recommendations

**Priority**: Critical fixes required for production release

---

## 🔥 PRIORITY 1: Fix Bulk RNA-seq Agent

### Issue: pyDESeq2 Integration Broken

**Problem**: Matplotlib compatibility issues preventing pyDESeq2 import
```
ImportError: cannot import name 'pyplot' from 'matplotlib'
```

**Solution Steps**:
1. **Update Dependencies**:
   ```bash
   # Add to pyproject.toml
   matplotlib = ">=3.8.0"
   pydeseq2 = ">=1.12.0"
   ```

2. **Fix Import Order**:
   ```python
   # In bulk_rnaseq_service.py
   import matplotlib
   matplotlib.use('Agg')  # Set backend before importing pyplot
   import matplotlib.pyplot as plt
   from pydeseq2 import DeseqDataSet
   ```

3. **Environment Configuration**:
   ```python
   # Add to service initialization
   os.environ['MPLBACKEND'] = 'Agg'
   ```

**Files to Modify**:
- `lobster/tools/bulk_rnaseq_service.py`
- `pyproject.toml`
- Test files in `tests/unit/tools/test_bulk_rnaseq_*`

---

### Issue: Pathway Enrichment Missing

**Problem**: Method throws "not implemented" error

**Solution Steps**:
1. **Add GSEApy Integration**:
   ```python
   import gseapy as gp
   from gseapy import enrichr

   def run_pathway_enrichment_analysis(self, modality_name: str,
                                     gene_list: List[str],
                                     gene_sets: str = 'GO_Biological_Process_2023'):
       """Implement pathway enrichment using GSEApy"""
       enr = enrichr(gene_list=gene_list,
                     gene_sets=gene_sets,
                     organism='Human',
                     outdir=self.results_dir)
       return enr.results
   ```

2. **Add Dependencies**:
   ```bash
   # Add to pyproject.toml
   gseapy = ">=1.0.4"
   ```

**Effort**: 1-2 days

---

### Issue: Test Suite Interface Misalignment

**Problem**: Tests expect methods that don't exist

**Solution Steps**:
1. **Update Method Signatures**:
   ```python
   # Fix in bulk_rnaseq_service.py
   def create_formula_design(self, formula: str, sample_data: pd.DataFrame):
       # Implement method that tests expect
       pass
   ```

2. **Remove Tests for Non-existent Methods**:
   - Remove tests for methods not in actual implementation
   - Update parameter expectations to match reality

3. **Add Missing Methods**:
   - Implement methods that tests reasonably expect
   - Or remove tests if methods are not needed

**Files to Fix**:
- `tests/unit/tools/test_bulk_rnaseq_service.py` (26 failing tests)
- `lobster/tools/bulk_rnaseq_service.py`

**Effort**: 2-3 days

---

## ⚠️ PRIORITY 2: Test Suite Alignment

### Fix Unit Test Failures

**Problems**:
- Test expectations don't match implementation
- Outdated mocking assumptions
- Missing validation functions in tests

**Solution Steps**:

1. **Update Agent Registry Tests**:
   ```python
   # Fix tests/unit/agents/test_agent_registry.py
   # Remove tests for validation functions that don't exist
   # Update return type expectations (objects vs strings)
   ```

2. **Fix Supervisor Tests**:
   ```python
   # Fix tests/unit/agents/test_supervisor.py
   # Update patch targets to match actual function locations
   # Remove tests for deprecated handoff functions
   ```

3. **Update Mock Expectations**:
   - Align mock return values with actual implementation
   - Update function signatures in mocks
   - Fix import paths in test files

**Effort**: 3-4 days

---

## 📋 PRIORITY 3: Enhance Implementation

### Complete Statistical Methods

**Improvements Needed**:
1. **Real DESeq2 Normalization**:
   ```python
   def deseq2_normalize(self, counts_df: pd.DataFrame):
       """Implement proper DESeq2 normalization"""
       dds = DeseqDataSet(counts=counts_df,
                         metadata=sample_metadata,
                         design_factors=design_factors)
       dds.deseq2()
       return dds.layers['normed_counts']
   ```

2. **Add Dispersion Estimation**:
   ```python
   def estimate_dispersion(self, dds: DeseqDataSet):
       """Proper dispersion estimation"""
       dds.fit_size_factors()
       dds.fit_genewise_dispersions()
       dds.fit_dispersion_trend()
       return dds
   ```

**Effort**: 3-5 days

---

### Add Integration Testing

**Missing Coverage**:
1. **End-to-End Workflow Tests**:
   ```python
   def test_complete_bulk_rnaseq_workflow():
       """Test data loading -> QC -> DE -> pathway"""
       # Implement comprehensive workflow test
   ```

2. **Agent Handoff Testing**:
   ```python
   def test_agent_handoffs():
       """Test supervisor -> bulk_rnaseq -> visualization"""
       # Test complete agent coordination
   ```

**Effort**: 2-3 days

---

## 🛠️ IMPLEMENTATION PLAN

### Week 1-2: Critical Fixes
- [ ] Fix pyDESeq2 integration
- [ ] Implement pathway enrichment
- [ ] Update failing test interfaces

### Week 3-4: Test Suite Updates
- [ ] Fix agent registry tests
- [ ] Update supervisor tests
- [ ] Align bulk RNA-seq service tests

### Week 5: Validation & Enhancement
- [ ] Complete statistical methods
- [ ] Add integration testing
- [ ] Performance validation

### Week 6: Final Validation
- [ ] End-to-end testing
- [ ] Documentation updates
- [ ] Release preparation

---

## 📊 SUCCESS METRICS

### Target Achievements:
- **Test Pass Rate**: >85% (currently 67%)
- **Bulk RNA-seq Functionality**: >90% (currently 60%)
- **Integration Coverage**: >80% (currently <50%)
- **Critical Features**: 100% implemented

### Validation Criteria:
- [ ] All pyDESeq2 tests passing
- [ ] Pathway enrichment working with real data
- [ ] Complete bulk RNA-seq workflow functional
- [ ] Agent handoffs working correctly
- [ ] Performance benchmarks met

---

## 🚀 AFTER FIXES - NEXT STEPS

### Short Term (1-2 months):
1. **Real Data Validation**: Test with authentic bulk RNA-seq datasets
2. **Performance Optimization**: Large dataset handling improvements
3. **Advanced Features**: Multi-factor experimental designs

### Medium Term (3-6 months):
1. **Multi-Omics Integration**: Cross-platform analysis workflows
2. **Cloud Deployment**: Production-ready cloud services
3. **External Tool Integration**: Connect with established pipelines

### Long Term (6+ months):
1. **Advanced Analytics**: Machine learning integration
2. **Visualization Enhancement**: Interactive analysis dashboards
3. **API Development**: External system integrations

---

**Timeline Summary**: 4-6 weeks to production readiness with focused effort on critical issues.

**Risk Mitigation**: Prioritize pyDESeq2 fix first as it's the highest impact blocker.