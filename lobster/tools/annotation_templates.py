"""
Annotation Templates for Manual Cell Type Annotation

This module provides predefined annotation templates for common tissue types,
including marker gene sets and hierarchical cell type definitions.

Templates include:
- PBMC (Peripheral Blood Mononuclear Cells)
- Brain tissue (cortex, hippocampus)
- Lung tissue
- Heart tissue
- Kidney tissue
- Liver tissue
- Intestinal tissue
- Skin tissue
- Tumor microenvironment
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class TissueType(Enum):
    """Enumeration of supported tissue types."""

    PBMC = "pbmc"
    BRAIN = "brain"
    LUNG = "lung"
    HEART = "heart"
    KIDNEY = "kidney"
    LIVER = "liver"
    INTESTINE = "intestine"
    SKIN = "skin"
    TUMOR = "tumor"
    CUSTOM = "custom"


@dataclass
class CellTypeTemplate:
    """Template for a specific cell type annotation."""

    name: str
    markers: List[str]
    category: str
    confidence_threshold: float = 0.5
    description: str = ""
    alternative_names: List[str] = None
    validation_status: str = (
        "preliminary"  # Options: "preliminary", "validated", "reference_based"
    )
    evidence_source: Optional[str] = (
        None  # Source of markers (e.g., "Azimuth_v0.4.6", "CellTypist_v2.0", "PanglaoDB", "Manual")
    )
    species: str = "human"  # Options: "human", "mouse", "multi"

    def __post_init__(self):
        if self.alternative_names is None:
            self.alternative_names = []


class AnnotationTemplateService:
    """
    Service for managing and applying predefined annotation templates.

    Provides tissue-specific cell type definitions with marker genes
    for automated and semi-automated annotation workflows.
    """

    def __init__(self):
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[TissueType, Dict[str, CellTypeTemplate]]:
        """Initialize all predefined templates."""

        templates = {}

        # PBMC Template
        templates[TissueType.PBMC] = self._create_pbmc_template()

        # Brain Template
        templates[TissueType.BRAIN] = self._create_brain_template()

        # Lung Template
        templates[TissueType.LUNG] = self._create_lung_template()

        # Heart Template
        templates[TissueType.HEART] = self._create_heart_template()

        # Kidney Template
        templates[TissueType.KIDNEY] = self._create_kidney_template()

        # Liver Template
        templates[TissueType.LIVER] = self._create_liver_template()

        # Intestine Template
        templates[TissueType.INTESTINE] = self._create_intestine_template()

        # Skin Template
        templates[TissueType.SKIN] = self._create_skin_template()

        # Tumor Microenvironment Template
        templates[TissueType.TUMOR] = self._create_tumor_template()

        return templates

    def _create_pbmc_template(self) -> Dict[str, CellTypeTemplate]:
        """Create PBMC (Peripheral Blood Mononuclear Cells) template."""

        return {
            "T cells CD4+": CellTypeTemplate(
                name="T cells CD4+",
                markers=["CD3D", "CD3E", "CD4", "IL7R", "CCR7", "LEF1"],
                category="T cells",
                confidence_threshold=0.6,
                description="CD4+ helper T cells",
                alternative_names=["CD4 T cells", "Helper T cells", "Th cells"],
            ),
            "T cells CD8+": CellTypeTemplate(
                name="T cells CD8+",
                markers=["CD3D", "CD3E", "CD8A", "CD8B", "CCL5", "GZMK"],
                category="T cells",
                confidence_threshold=0.6,
                description="CD8+ cytotoxic T cells",
                alternative_names=["CD8 T cells", "Cytotoxic T cells", "CTL"],
            ),
            "T cells regulatory": CellTypeTemplate(
                name="T cells regulatory",
                markers=["CD3D", "CD3E", "CD4", "FOXP3", "IL2RA", "CTLA4"],
                category="T cells",
                confidence_threshold=0.7,
                description="Regulatory T cells (Tregs)",
                alternative_names=["Tregs", "Regulatory T cells", "FOXP3+ T cells"],
            ),
            "T cells gamma delta": CellTypeTemplate(
                name="T cells gamma delta",
                markers=["TRGC1", "TRGC2", "TRDV2", "CD3D", "CD3G"],
                category="T cells",
                confidence_threshold=0.7,
                description="Gamma delta T cells",
                alternative_names=["γδ T cells", "Gamma delta T cells"],
            ),
            "T cells exhausted": CellTypeTemplate(
                name="T cells exhausted",
                markers=["CD3D", "PDCD1", "CTLA4", "LAG3", "TIGIT", "HAVCR2", "CD57"],
                category="T cells",
                confidence_threshold=0.7,
                description="Exhausted/senescent T cells (immunosenescence marker)",
                alternative_names=[
                    "Senescent T cells",
                    "Immunosenescent T cells",
                    "PD1+ T cells",
                ],
                validation_status="validated",
                evidence_source="Multiple aging studies - robust RNA markers",
                species="human",
            ),
            "NK cells": CellTypeTemplate(
                name="NK cells",
                markers=["GNLY", "NKG7", "KLRD1", "NCR1", "PRF1", "GZMB"],
                category="NK cells",
                confidence_threshold=0.6,
                description="Natural killer cells",
                alternative_names=[
                    "Natural killer cells",
                    "Large granular lymphocytes",
                ],
            ),
            "NK cells aged": CellTypeTemplate(
                name="NK cells aged",
                markers=["GNLY", "NKG7", "CD57", "KLRG1", "FCGR3A", "PRF1"],
                category="NK cells",
                confidence_threshold=0.7,
                description="Aged/mature NK cells with reduced proliferative capacity",
                alternative_names=[
                    "CD57+ NK cells",
                    "Mature NK cells",
                    "Senescent NK cells",
                ],
                validation_status="validated",
                evidence_source="Immunosenescence studies",
                species="human",
            ),
            "B cells naive": CellTypeTemplate(
                name="B cells naive",
                markers=["CD19", "MS4A1", "CD79A", "CD79B", "IGHD", "TCL1A"],
                category="B cells",
                confidence_threshold=0.6,
                description="Naive B cells",
                alternative_names=["Naive B cells", "Mature B cells"],
            ),
            "B cells memory": CellTypeTemplate(
                name="B cells memory",
                markers=["CD19", "MS4A1", "CD79A", "CD27", "TNFRSF13B", "IGHG1"],
                category="B cells",
                confidence_threshold=0.6,
                description="Memory B cells",
                alternative_names=["Memory B cells", "Switched memory B cells"],
            ),
            "Plasma cells": CellTypeTemplate(
                name="Plasma cells",
                markers=["IGHG1", "IGHG2", "IGHG3", "IGHG4", "JCHAIN", "XBP1"],
                category="B cells",
                confidence_threshold=0.7,
                description="Antibody-secreting plasma cells",
                alternative_names=["Plasmablasts", "Antibody-secreting cells"],
            ),
            "Monocytes CD14+": CellTypeTemplate(
                name="Monocytes CD14+",
                markers=["CD14", "LYZ", "S100A8", "S100A9", "FCN1", "VCAN"],
                category="Myeloid",
                confidence_threshold=0.6,
                description="Classical CD14+ monocytes",
                alternative_names=["Classical monocytes", "CD14+ monocytes"],
            ),
            "Monocytes inflammatory": CellTypeTemplate(
                name="Monocytes inflammatory",
                markers=["CD14", "IL1B", "IL6", "TNF", "NLRP3", "CXCL8", "CCL2"],
                category="Myeloid",
                confidence_threshold=0.7,
                description="Inflammatory monocytes with high cytokine expression (inflammaging)",
                alternative_names=[
                    "Inflammaging monocytes",
                    "Cytokine-high monocytes",
                    "Activated monocytes",
                ],
                validation_status="preliminary",
                evidence_source="Nature Aging 2021 (PMID: 33564145)",
                species="human",
            ),
            "Monocytes CD16+": CellTypeTemplate(
                name="Monocytes CD16+",
                markers=["FCGR3A", "MS4A7", "LST1", "AIF1", "SERPINA1"],
                category="Myeloid",
                confidence_threshold=0.6,
                description="Non-classical CD16+ monocytes",
                alternative_names=[
                    "Non-classical monocytes",
                    "CD16+ monocytes",
                    "FCGR3A+ monocytes",
                ],
            ),
            "Dendritic cells cDC1": CellTypeTemplate(
                name="Dendritic cells cDC1",
                markers=["CLEC9A", "XCR1", "BATF3", "IRF8"],
                category="Myeloid",
                confidence_threshold=0.7,
                description="Conventional type 1 dendritic cells",
                alternative_names=["cDC1", "Type 1 DC", "CLEC9A+ DC"],
            ),
            "Dendritic cells cDC2": CellTypeTemplate(
                name="Dendritic cells cDC2",
                markers=["CD1C", "FCER1A", "ITGAX", "IRF4"],
                category="Myeloid",
                confidence_threshold=0.7,
                description="Conventional type 2 dendritic cells",
                alternative_names=["cDC2", "Type 2 DC", "CD1C+ DC"],
            ),
            "Plasmacytoid dendritic cells": CellTypeTemplate(
                name="Plasmacytoid dendritic cells",
                markers=["IL3RA", "CLEC4C", "TCF4", "GZMB"],
                category="Myeloid",
                confidence_threshold=0.7,
                description="Plasmacytoid dendritic cells",
                alternative_names=["pDC", "Plasmacytoid DC", "Type I interferon DC"],
            ),
            "Platelets": CellTypeTemplate(
                name="Platelets",
                markers=["PPBP", "PF4", "GP9", "GP1BA", "ITGA2B", "TUBB1"],
                category="Platelets",
                confidence_threshold=0.8,
                description="Blood platelets",
                alternative_names=["Thrombocytes", "Platelet-derived particles"],
            ),
            # NOTE: "Debris" template removed - QC metrics (high_mt, low_genes, low_counts)
            # are not gene markers and belong in QC pipeline, not annotation templates.
            # Use quality_service.py for QC filtering instead.
        }

    def _create_brain_template(self) -> Dict[str, CellTypeTemplate]:
        """Create brain tissue template."""

        return {
            "Excitatory neurons": CellTypeTemplate(
                name="Excitatory neurons",
                markers=["SLC17A7", "CAMK2A", "RBFOX3", "NEUROD2", "NEUROD6", "SATB2"],
                category="Neurons",
                confidence_threshold=0.6,
                description="Glutamatergic excitatory neurons",
                alternative_names=["Glutamatergic neurons", "Pyramidal neurons"],
            ),
            "Inhibitory neurons": CellTypeTemplate(
                name="Inhibitory neurons",
                markers=["GAD1", "GAD2", "SLC32A1", "PVALB", "SST", "VIP"],
                category="Neurons",
                confidence_threshold=0.6,
                description="GABAergic inhibitory neurons",
                alternative_names=["GABAergic neurons", "Interneurons"],
            ),
            "Astrocytes": CellTypeTemplate(
                name="Astrocytes",
                markers=["GFAP", "AQP4", "ALDH1L1", "S100B", "SOX9", "SLC1A3"],
                category="Glia",
                confidence_threshold=0.6,
                description="Astrocytic glial cells",
                alternative_names=["Astroglia", "Fibrous astrocytes"],
            ),
            "Oligodendrocytes": CellTypeTemplate(
                name="Oligodendrocytes",
                markers=["MBP", "MOG", "PLP1", "MAG", "CNP", "MOBP"],
                category="Glia",
                confidence_threshold=0.6,
                description="Myelinating oligodendrocytes",
                alternative_names=["Mature oligodendrocytes", "Myelinating glia"],
            ),
            "Oligodendrocyte precursors": CellTypeTemplate(
                name="Oligodendrocyte precursors",
                markers=["PDGFRA", "CSPG4", "SOX10", "OLIG2", "OLIG1", "NKX2-2"],
                category="Glia",
                confidence_threshold=0.6,
                description="Oligodendrocyte precursor cells (OPCs)",
                alternative_names=["OPC", "NG2 cells", "Polydendrocytes"],
            ),
            "Microglia": CellTypeTemplate(
                name="Microglia",
                markers=["CX3CR1", "P2RY12", "TMEM119", "AIF1", "CSF1R", "TREM2"],
                category="Immune",
                confidence_threshold=0.7,
                description="Brain-resident immune cells",
                alternative_names=["Brain macrophages", "Resident microglia"],
            ),
            "Endothelial cells": CellTypeTemplate(
                name="Endothelial cells",
                markers=["PECAM1", "CLDN5", "KDR", "FLT1", "ABCB1", "CDH5"],
                category="Vascular",
                confidence_threshold=0.6,
                description="Blood-brain barrier endothelial cells",
                alternative_names=["Brain endothelium", "BBB endothelial cells"],
            ),
            "Pericytes": CellTypeTemplate(
                name="Pericytes",
                markers=["PDGFRB", "RGS5", "CSPG4", "KCNJ8", "ABCC9", "ANPEP"],
                category="Vascular",
                confidence_threshold=0.6,
                description="Vascular mural cells",
                alternative_names=["Vascular pericytes", "Mural cells"],
            ),
        }

    def _create_lung_template(self) -> Dict[str, CellTypeTemplate]:
        """Create lung tissue template."""

        return {
            "Alveolar epithelial type I": CellTypeTemplate(
                name="Alveolar epithelial type I",
                markers=["AGER", "CAV1", "PDPN", "HOPX", "EMP2", "CLIC5"],
                category="Epithelial",
                confidence_threshold=0.6,
                description="Alveolar epithelial type I cells",
                alternative_names=["AT1 cells", "Pneumocytes type I"],
            ),
            "Alveolar epithelial type II": CellTypeTemplate(
                name="Alveolar epithelial type II",
                markers=[
                    "SFTPA1",
                    "SFTPA2",
                    "SFTPB",
                    "SFTPC",
                    "SFTPD",
                    "ABCA3",
                    "SLC34A2",
                    "LPCAT1",
                ],
                category="Epithelial",
                confidence_threshold=0.6,
                description="Alveolar epithelial type II cells",
                alternative_names=["AT2 cells", "Pneumocytes type II"],
            ),
            "Club cells": CellTypeTemplate(
                name="Club cells",
                markers=["SCGB1A1", "SCGB3A1", "KRT13", "UGT1A6"],
                category="Epithelial",
                confidence_threshold=0.6,
                description="Club cells (Clara cells)",
                alternative_names=["Clara cells", "Non-ciliated bronchiolar cells"],
            ),
            "Ciliated cells": CellTypeTemplate(
                name="Ciliated cells",
                markers=["FOXJ1", "DNAH5", "CCDC40", "RSPH1", "TPPP3", "CETN2"],
                category="Epithelial",
                confidence_threshold=0.6,
                description="Ciliated epithelial cells",
                alternative_names=["Multiciliated cells", "Motile ciliated cells"],
            ),
            "Basal cells": CellTypeTemplate(
                name="Basal cells",
                markers=["TP63", "KRT5", "KRT14", "NGFR", "MYC", "DLX3"],
                category="Epithelial",
                confidence_threshold=0.6,
                description="Airway basal stem cells",
                alternative_names=["Airway basal cells", "Progenitor cells"],
            ),
            "Alveolar macrophages": CellTypeTemplate(
                name="Alveolar macrophages",
                markers=["MARCO", "PPARG", "MRC1", "SIGLEC1", "C1QA", "C1QB", "APOE"],
                category="Immune",
                confidence_threshold=0.6,
                description="Tissue-resident alveolar macrophages",
                alternative_names=["AM", "Lung macrophages"],
            ),
            "Interstitial macrophages": CellTypeTemplate(
                name="Interstitial macrophages",
                markers=["CD68", "CD163", "MRC1", "LYVE1", "F13A1", "FOLR2"],
                category="Immune",
                confidence_threshold=0.6,
                description="Interstitial tissue macrophages",
                alternative_names=["IM", "Tissue macrophages"],
            ),
            "Pulmonary capillary endothelium": CellTypeTemplate(
                name="Pulmonary capillary endothelium",
                markers=["PECAM1", "VWF", "CDH5", "CA4", "EDNRB", "GPIHBP1"],
                category="Vascular",
                confidence_threshold=0.6,
                description="Pulmonary capillary endothelial cells",
                alternative_names=["Lung capillary EC", "Pulmonary EC"],
            ),
        }

    def _create_heart_template(self) -> Dict[str, CellTypeTemplate]:
        """Create heart tissue template."""

        return {
            "Cardiomyocytes": CellTypeTemplate(
                name="Cardiomyocytes",
                markers=["MYH6", "MYH7", "TNNT2", "TNNI3", "ACTC1", "MYL2"],
                category="Muscle",
                confidence_threshold=0.7,
                description="Cardiac muscle cells",
                alternative_names=["Cardiac myocytes", "Heart muscle cells"],
            ),
            "Cardiac fibroblasts": CellTypeTemplate(
                name="Cardiac fibroblasts",
                markers=["COL1A1", "COL3A1", "DCN", "LUM", "POSTN", "TCF21"],
                category="Stromal",
                confidence_threshold=0.6,
                description="Cardiac stromal fibroblasts",
                alternative_names=["Heart fibroblasts", "Cardiac stromal cells"],
            ),
            "Smooth muscle cells": CellTypeTemplate(
                name="Smooth muscle cells",
                markers=["ACTA2", "TAGLN", "MYH11", "CNN1", "MYLK", "CALD1"],
                category="Muscle",
                confidence_threshold=0.6,
                description="Vascular smooth muscle cells",
                alternative_names=["VSMC", "Vascular SMC"],
            ),
            "Endothelial cells": CellTypeTemplate(
                name="Endothelial cells",
                markers=["PECAM1", "VWF", "CDH5", "KDR", "FLT1", "PLVAP"],
                category="Vascular",
                confidence_threshold=0.6,
                description="Cardiac endothelial cells",
                alternative_names=["Cardiac EC", "Heart endothelium"],
            ),
            "Cardiac macrophages": CellTypeTemplate(
                name="Cardiac macrophages",
                markers=["CD68", "CD163", "LYVE1", "MRC1", "C1QA", "FOLR2"],
                category="Immune",
                confidence_threshold=0.6,
                description="Heart-resident macrophages",
                alternative_names=["Heart macrophages", "Cardiac immune cells"],
            ),
        }

    def _create_kidney_template(self) -> Dict[str, CellTypeTemplate]:
        """Create kidney tissue template."""

        return {
            "Podocytes": CellTypeTemplate(
                name="Podocytes",
                markers=["NPHS1", "NPHS2", "PODXL", "WT1", "SYNPO", "MAGI2"],
                category="Epithelial",
                confidence_threshold=0.7,
                description="Glomerular visceral epithelial cells",
                alternative_names=[
                    "Glomerular epithelium",
                    "Visceral epithelial cells",
                ],
            ),
            "Proximal tubule": CellTypeTemplate(
                name="Proximal tubule",
                markers=["LRP2", "SLC34A1", "SLC5A2", "CUBN", "SLC22A8", "SLC5A12"],
                category="Epithelial",
                confidence_threshold=0.6,
                description="Proximal tubule epithelial cells",
                alternative_names=["PT cells", "Proximal tubular epithelium"],
            ),
            "Distal tubule": CellTypeTemplate(
                name="Distal tubule",
                markers=["SLC12A3", "PVALB", "TRPM6", "CALB1", "EGF", "WNK4"],
                category="Epithelial",
                confidence_threshold=0.6,
                description="Distal convoluted tubule cells",
                alternative_names=["DCT cells", "Distal convoluted tubule"],
            ),
            "Collecting duct principal": CellTypeTemplate(
                name="Collecting duct principal",
                markers=["AQP2", "AQP3", "SCNN1G", "AVPR2", "KCNJ10"],
                category="Epithelial",
                confidence_threshold=0.6,
                description="Collecting duct principal cells",
                alternative_names=["CD principal cells", "Principal cells"],
            ),
            "Collecting duct intercalated alpha": CellTypeTemplate(
                name="Collecting duct intercalated alpha",
                markers=["SLC4A1", "ATP6V1B1", "ATP6V0D2"],
                category="Epithelial",
                confidence_threshold=0.6,
                description="Collecting duct type A intercalated cells",
                alternative_names=["IC-A cells", "Alpha intercalated cells"],
            ),
            "Collecting duct intercalated beta": CellTypeTemplate(
                name="Collecting duct intercalated beta",
                markers=["SLC26A4", "FOXI1", "ATP6V1B1"],
                category="Epithelial",
                confidence_threshold=0.6,
                description="Collecting duct type B intercalated cells",
                alternative_names=["IC-B cells", "Beta intercalated cells"],
            ),
            "Loop of Henle": CellTypeTemplate(
                name="Loop of Henle",
                markers=["SLC12A1", "UMOD", "PTGER3", "CLDN16", "KCNJ1", "SLC2A9"],
                category="Epithelial",
                confidence_threshold=0.6,
                description="Loop of Henle epithelial cells",
                alternative_names=["Thick ascending limb", "TAL cells"],
            ),
            "Glomerular endothelium": CellTypeTemplate(
                name="Glomerular endothelium",
                markers=["PECAM1", "KDR", "PTPRB", "EMCN", "CLDN5", "PLVAP"],
                category="Vascular",
                confidence_threshold=0.6,
                description="Glomerular capillary endothelium",
                alternative_names=["Glomerular EC", "Fenestrated endothelium"],
            ),
        }

    def _create_liver_template(self) -> Dict[str, CellTypeTemplate]:
        """Create liver tissue template."""

        return {
            "Hepatocytes": CellTypeTemplate(
                name="Hepatocytes",
                markers=["ALB", "APOB", "CYP3A4", "G6PC", "PCK1", "CPS1"],
                category="Hepatic",
                confidence_threshold=0.7,
                description="Liver parenchymal cells",
                alternative_names=["Liver cells", "Hepatic parenchyma"],
            ),
            "Cholangiocytes": CellTypeTemplate(
                name="Cholangiocytes",
                markers=["KRT19", "KRT7", "SOX9", "CFTR", "AE2", "EPCAM"],
                category="Epithelial",
                confidence_threshold=0.6,
                description="Bile duct epithelial cells",
                alternative_names=["Bile duct cells", "Biliary epithelium"],
            ),
            "Hepatic stellate cells": CellTypeTemplate(
                name="Hepatic stellate cells",
                markers=["ACTA2", "COL1A1", "PDGFRB", "RGS5", "LRAT", "VIM"],
                category="Stromal",
                confidence_threshold=0.6,
                description="Liver perisinusoidal cells",
                alternative_names=["HSC", "Ito cells", "Perisinusoidal cells"],
            ),
            "Kupffer cells": CellTypeTemplate(
                name="Kupffer cells",
                markers=["CD68", "CLEC4F", "VSIG4", "TIMD4", "C1QA", "MARCO"],
                category="Immune",
                confidence_threshold=0.7,
                description="Liver-resident macrophages",
                alternative_names=["Liver macrophages", "Hepatic macrophages"],
            ),
            "Liver sinusoidal endothelium": CellTypeTemplate(
                name="Liver sinusoidal endothelium",
                markers=["PECAM1", "STAB2", "LYVE1", "FCN2", "OIT3", "CLEC4G"],
                category="Vascular",
                confidence_threshold=0.6,
                description="Hepatic sinusoidal endothelial cells",
                alternative_names=["LSEC", "Sinusoidal EC"],
            ),
        }

    def _create_intestine_template(self) -> Dict[str, CellTypeTemplate]:
        """Create intestinal tissue template."""

        return {
            "Enterocytes": CellTypeTemplate(
                name="Enterocytes",
                markers=["FABP2", "APOA1", "APOB", "SI", "DPP4", "ALPI"],
                category="Epithelial",
                confidence_threshold=0.6,
                description="Absorptive intestinal epithelial cells",
                alternative_names=["Absorptive cells", "Intestinal epithelium"],
            ),
            "Goblet cells": CellTypeTemplate(
                name="Goblet cells",
                markers=["MUC2", "TFF3", "CLCA1", "FCGBP", "ZG16", "AGR2"],
                category="Epithelial",
                confidence_threshold=0.7,
                description="Mucus-secreting epithelial cells",
                alternative_names=["Mucin-producing cells", "Secretory cells"],
            ),
            "Paneth cells": CellTypeTemplate(
                name="Paneth cells",
                markers=["LYZ", "DEFA5", "DEFA6", "PLA2G2A", "ITLN1", "REG3A"],
                category="Epithelial",
                confidence_threshold=0.7,
                description="Antimicrobial peptide-secreting cells",
                alternative_names=["Antimicrobial cells", "Crypt base cells"],
            ),
            "Enteroendocrine cells": CellTypeTemplate(
                name="Enteroendocrine cells",
                markers=["CHGA", "CHGB", "GCG", "CCK", "GIP", "NEUROD1"],
                category="Epithelial",
                confidence_threshold=0.7,
                description="Hormone-secreting epithelial cells",
                alternative_names=["EEC", "Hormone-producing cells"],
            ),
            "Intestinal stem cells": CellTypeTemplate(
                name="Intestinal stem cells",
                markers=["LGR5", "ASCL2", "OLFM4", "MSI1", "EPHB2", "CD44"],
                category="Stem",
                confidence_threshold=0.7,
                description="Crypt base columnar stem cells",
                alternative_names=["CBC cells", "ISC", "Lgr5+ cells"],
            ),
        }

    def _create_skin_template(self) -> Dict[str, CellTypeTemplate]:
        """Create skin tissue template."""

        return {
            "Keratinocytes basal": CellTypeTemplate(
                name="Keratinocytes basal",
                markers=["KRT14", "KRT5", "TP63", "COL17A1", "ITGB1", "LAMA3"],
                category="Epithelial",
                confidence_threshold=0.6,
                description="Basal layer keratinocytes",
                alternative_names=["Basal keratinocytes", "Basal cells"],
            ),
            "Keratinocytes spinous": CellTypeTemplate(
                name="Keratinocytes spinous",
                markers=["KRT1", "KRT10", "DSG1", "PKP1", "GRHL1", "OVOL1"],
                category="Epithelial",
                confidence_threshold=0.6,
                description="Spinous layer keratinocytes",
                alternative_names=["Spinous keratinocytes", "Suprabasal cells"],
            ),
            "Melanocytes": CellTypeTemplate(
                name="Melanocytes",
                markers=["MLANA", "TYR", "TYRP1", "DCT", "MITF", "PMEL"],
                category="Pigment",
                confidence_threshold=0.7,
                description="Pigment-producing cells",
                alternative_names=["Pigment cells", "Melanin-producing cells"],
            ),
            "Fibroblasts": CellTypeTemplate(
                name="Fibroblasts",
                markers=["COL1A1", "COL3A1", "VIM", "DCN", "LUM", "PDGFRA"],
                category="Stromal",
                confidence_threshold=0.6,
                description="Dermal fibroblasts",
                alternative_names=["Dermal fibroblasts", "Stromal cells"],
            ),
            "Langerhans cells": CellTypeTemplate(
                name="Langerhans cells",
                markers=["CD207", "CD1A", "HLA-DRA", "ITGAX", "FCER1G"],
                category="Immune",
                confidence_threshold=0.7,
                description="Epidermal dendritic cells",
                alternative_names=["Epidermal DC", "LC"],
            ),
        }

    def _create_tumor_template(self) -> Dict[str, CellTypeTemplate]:
        """Create tumor microenvironment template."""

        return {
            # WARNING: "Tumor cells" detection via proliferation markers is NOT RECOMMENDED
            # This approach will:
            #   1. Mislabel proliferating immune/stromal cells as "tumor"
            #   2. Miss non-cycling malignant clones (often the most dangerous)
            # RECOMMENDED: Use CNV inference (inferCNV, CopyKAT) to identify malignant cells
            # Then annotate malignant lineage with tissue-specific markers:
            #   - Carcinoma: EPCAM, KRT8, KRT18, KRT19
            #   - Glioma: OLIG2, PDGFRA
            #   - Sarcoma: VIM, COL1A1 (with CNV evidence)
            # Keep this template ONLY for backward compatibility - prefer CNV-based detection
            "Tumor cells": CellTypeTemplate(
                name="Tumor cells",
                markers=["MKI67", "PCNA", "TOP2A", "CCNA2", "BIRC5", "CDK1"],
                category="Malignant",
                confidence_threshold=0.5,
                description="Proliferating cells (WARNING: not reliable for tumor detection - use CNV inference)",
                alternative_names=[
                    "Cancer cells",
                    "Malignant cells",
                    "Neoplastic cells",
                ],
                validation_status="deprecated",
                evidence_source="Proliferation markers only - NOT tumor-specific",
            ),
            "T cells exhausted": CellTypeTemplate(
                name="T cells exhausted",
                markers=["PDCD1", "CTLA4", "HAVCR2", "LAG3", "TIGIT", "TOX"],
                category="Immune",
                confidence_threshold=0.6,
                description="Exhausted/dysfunctional T cells",
                alternative_names=["Exhausted T cells", "Dysfunctional T cells"],
            ),
            "Tumor-associated macrophages M1": CellTypeTemplate(
                name="Tumor-associated macrophages M1",
                markers=["CD68", "CD80", "CD86", "IL1B", "TNF", "NOS2"],
                category="Immune",
                confidence_threshold=0.6,
                description="M1-polarized tumor-associated macrophages",
                alternative_names=["M1 TAM", "Pro-inflammatory macrophages"],
            ),
            "Tumor-associated macrophages M2": CellTypeTemplate(
                name="Tumor-associated macrophages M2",
                markers=["CD68", "CD163", "MRC1", "ARG1", "IL10", "TGFB1"],
                category="Immune",
                confidence_threshold=0.6,
                description="M2-polarized tumor-associated macrophages",
                alternative_names=["M2 TAM", "Anti-inflammatory macrophages"],
            ),
            "Cancer-associated fibroblasts": CellTypeTemplate(
                name="Cancer-associated fibroblasts",
                markers=["COL1A1", "ACTA2", "FAP", "PDGFRA", "PDGFRB", "POSTN"],
                category="Stromal",
                confidence_threshold=0.6,
                description="Activated stromal fibroblasts in tumor",
                alternative_names=["CAF", "Myofibroblasts", "Activated fibroblasts"],
            ),
            "Tumor endothelium": CellTypeTemplate(
                name="Tumor endothelium",
                markers=["PECAM1", "VWF", "VEGFA", "ANGPT2", "FLT1", "KDR"],
                category="Vascular",
                confidence_threshold=0.6,
                description="Tumor-associated endothelial cells",
                alternative_names=["Tumor EC", "Angiogenic endothelium"],
            ),
            "Regulatory T cells": CellTypeTemplate(
                name="Regulatory T cells",
                markers=["CD3D", "CD4", "FOXP3", "IL2RA", "CTLA4", "TNFRSF18"],
                category="Immune",
                confidence_threshold=0.7,
                description="Immunosuppressive regulatory T cells",
                alternative_names=["Tregs", "Suppressor T cells"],
            ),
            # NOTE: SASP/Senescence templates removed (v0.1.0)
            # Reason: Not RNA-robust - markers like GLB1 (SA-β-gal), TP53, RB1 confuse
            # quiescence vs senescence. Requires protein-level assays or functional validation.
            # For senescence detection, use: (1) custom validated markers for your context,
            # (2) multi-marker scoring with lineage-specific SASP panels, (3) orthogonal
            # validation (SA-β-gal staining, cell cycle analysis, secretome profiling).
        }

    def get_template(
        self, tissue_type: TissueType
    ) -> Optional[Dict[str, CellTypeTemplate]]:
        """
        Get annotation template for specified tissue type.

        Args:
            tissue_type: Type of tissue to get template for

        Returns:
            Dictionary of cell type templates or None if not found
        """
        return self.templates.get(tissue_type)

    def get_all_tissue_types(self) -> List[TissueType]:
        """Get list of all available tissue types."""
        return list(self.templates.keys())

    def get_cell_types_for_tissue(self, tissue_type: TissueType) -> List[str]:
        """
        Get list of cell type names for a tissue type.

        Args:
            tissue_type: Type of tissue

        Returns:
            List of cell type names
        """
        template = self.get_template(tissue_type)
        return list(template.keys()) if template else []

    def get_markers_for_cell_type(
        self, tissue_type: TissueType, cell_type: str
    ) -> List[str]:
        """
        Get marker genes for a specific cell type.

        Args:
            tissue_type: Type of tissue
            cell_type: Name of cell type

        Returns:
            List of marker gene names
        """
        template = self.get_template(tissue_type)
        if template and cell_type in template:
            return template[cell_type].markers
        return []

    def search_cell_types_by_marker(
        self, marker_gene: str, tissue_type: Optional[TissueType] = None
    ) -> List[Tuple[TissueType, str]]:
        """
        Search for cell types that express a specific marker gene.

        Args:
            marker_gene: Gene symbol to search for
            tissue_type: Optional tissue type to limit search

        Returns:
            List of (tissue_type, cell_type) tuples
        """
        results = []

        tissues_to_search = [tissue_type] if tissue_type else self.templates.keys()

        for tissue in tissues_to_search:
            template = self.templates[tissue]
            for cell_type_name, cell_type_template in template.items():
                if marker_gene.upper() in [
                    marker.upper() for marker in cell_type_template.markers
                ]:
                    results.append((tissue, cell_type_name))

        return results

    def create_custom_template(
        self, name: str, cell_types: Dict[str, CellTypeTemplate]
    ) -> None:
        """
        Create a custom annotation template.

        Args:
            name: Name for the custom template
            cell_types: Dictionary of cell type templates
        """
        # Create custom tissue type
        custom_tissue = TissueType.CUSTOM
        if name not in [t.value for t in TissueType]:
            # For now, use CUSTOM enum - in a full implementation,
            # we might want to dynamically create new enum values
            pass

        self.templates[custom_tissue] = cell_types

    def apply_template_to_clusters(
        self,
        adata,
        tissue_type: TissueType,
        cluster_col: str = "leiden",
        expression_threshold: float = 0.5,
    ) -> Dict[str, str]:
        """
        Apply template to automatically suggest cell types for clusters.

        Args:
            adata: AnnData object with expression data
            tissue_type: Tissue type template to use
            cluster_col: Column containing cluster assignments
            expression_threshold: Minimum expression level for marker detection

        Returns:
            Dictionary mapping cluster IDs to suggested cell types
        """
        template = self.get_template(tissue_type)
        if not template:
            return {}

        cluster_suggestions = {}
        unique_clusters = adata.obs[cluster_col].unique()

        for cluster_id in unique_clusters:
            cluster_mask = adata.obs[cluster_col] == cluster_id
            cluster_cells = adata[cluster_mask]

            # Calculate scores for each cell type
            cell_type_scores = {}

            for cell_type_name, cell_type_template in template.items():
                score = self._calculate_cell_type_score(
                    cluster_cells, cell_type_template, expression_threshold
                )
                cell_type_scores[cell_type_name] = score

            # Get best matching cell type
            if cell_type_scores:
                best_cell_type = max(cell_type_scores, key=cell_type_scores.get)
                best_score = cell_type_scores[best_cell_type]

                # Only suggest if score meets confidence threshold
                template_obj = template[best_cell_type]
                if best_score >= template_obj.confidence_threshold:
                    cluster_suggestions[str(cluster_id)] = best_cell_type
                else:
                    cluster_suggestions[str(cluster_id)] = "Unknown"

        return cluster_suggestions

    def _calculate_cell_type_score(
        self,
        cluster_data,
        cell_type_template: CellTypeTemplate,
        expression_threshold: float,
    ) -> float:
        """
        Calculate how well a cluster matches a cell type template.

        Args:
            cluster_data: AnnData object for cluster cells
            cell_type_template: Cell type template to match against
            expression_threshold: Minimum expression level

        Returns:
            Matching score between 0 and 1
        """
        available_markers = [
            marker
            for marker in cell_type_template.markers
            if marker in cluster_data.var_names
        ]

        if not available_markers:
            return 0.0

        # Calculate mean expression of marker genes in cluster
        marker_expressions = []

        for marker in available_markers:
            marker_idx = cluster_data.var_names.get_loc(marker)
            if hasattr(cluster_data.X, "toarray"):
                # Sparse matrix
                marker_expr = cluster_data.X[:, marker_idx].toarray().flatten()
            else:
                # Dense matrix
                marker_expr = cluster_data.X[:, marker_idx]

            # Calculate percentage of cells expressing marker above threshold
            pct_expressing = np.sum(marker_expr > expression_threshold) / len(
                marker_expr
            )
            marker_expressions.append(pct_expressing)

        # Return mean percentage of cells expressing markers
        return np.mean(marker_expressions)

    def export_template(
        self, tissue_type: TissueType, output_path: str, format: str = "json"
    ) -> None:
        """
        Export annotation template to file.

        Args:
            tissue_type: Tissue type to export
            output_path: Path to save file
            format: Export format ('json' or 'csv')
        """
        template = self.get_template(tissue_type)
        if not template:
            raise ValueError(f"Template for {tissue_type} not found")

        if format.lower() == "json":
            import json

            # Convert template to serializable format
            export_data = {"tissue_type": tissue_type.value, "cell_types": {}}

            for cell_type_name, cell_type_template in template.items():
                export_data["cell_types"][cell_type_name] = {
                    "name": cell_type_template.name,
                    "markers": cell_type_template.markers,
                    "category": cell_type_template.category,
                    "confidence_threshold": cell_type_template.confidence_threshold,
                    "description": cell_type_template.description,
                    "alternative_names": cell_type_template.alternative_names,
                }

            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)

        elif format.lower() == "csv":
            import pandas as pd

            # Convert to flat CSV format
            rows = []
            for cell_type_name, cell_type_template in template.items():
                rows.append(
                    {
                        "tissue_type": tissue_type.value,
                        "cell_type": cell_type_template.name,
                        "category": cell_type_template.category,
                        "markers": ", ".join(cell_type_template.markers),
                        "confidence_threshold": cell_type_template.confidence_threshold,
                        "description": cell_type_template.description,
                        "alternative_names": ", ".join(
                            cell_type_template.alternative_names
                        ),
                    }
                )

            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def import_template(self, input_path: str, format: str = "json") -> TissueType:
        """
        Import annotation template from file.

        Args:
            input_path: Path to template file
            format: Import format ('json' or 'csv')

        Returns:
            TissueType of imported template
        """
        if format.lower() == "json":
            import json

            with open(input_path, "r") as f:
                import_data = json.load(f)

            tissue_type = TissueType(import_data["tissue_type"])
            cell_types = {}

            for cell_type_name, cell_type_data in import_data["cell_types"].items():
                cell_types[cell_type_name] = CellTypeTemplate(
                    name=cell_type_data["name"],
                    markers=cell_type_data["markers"],
                    category=cell_type_data["category"],
                    confidence_threshold=cell_type_data.get(
                        "confidence_threshold", 0.5
                    ),
                    description=cell_type_data.get("description", ""),
                    alternative_names=cell_type_data.get("alternative_names", []),
                )

            self.templates[tissue_type] = cell_types
            return tissue_type

        else:
            raise ValueError(f"Unsupported import format: {format}")
