from typing import Dict, Optional
import atexit
import sys


# Endpoint descriptions are defined immediately below (bundled from former endpoint_desc.py).

# -----------------------------------------------------------------------------
# Endpoint descriptions (bundled; formerly MolDeTox/endpoint_desc.py)
# -----------------------------------------------------------------------------

def get_dataset_context(dataset_name: Optional[str] = None, endpoint: Optional[str] = None) -> str:
    """Context sentence describing what the dataset's endpoint measures, prepended to the question.

    Accepts either ('tox21', 'NR-AR') or the combined form ('tox21_NR-AR'). Returns an empty
    string for datasets that need no context.
    """
    if not dataset_name and not endpoint:
        return ""
    
    # Handle cases where endpoint contains dataset name (e.g., 'tox21_NR-AR', 'herg', 'ames')
    if endpoint:
        endpoint_lower = endpoint.lower()
        
        # Check if endpoint starts with dataset prefix
        if endpoint_lower.startswith('tox21_'):
            # Extract actual endpoint name
            actual_endpoint = endpoint.replace('tox21_', '')
            return get_tox21_endpoint_description(actual_endpoint)
        
        # Check if endpoint is a SIDER endpoint
        sider_endpoints = [
            "blood and lymphatic system disorders", "cardiac disorders",
            "congenital, familial and genetic disorders", "ear and labyrinth disorders",
            "eye disorders", "general disorders and administration site conditions",
            "hepatobiliary disorders", "immune system disorders",
            "infections and infestations", "injury, poisoning and procedural complications",
            "investigations", "metabolism and nutrition disorders",
            "musculoskeletal and connective tissue disorders",
            "neoplasms benign, malignant and unspecified (incl cysts and polyps)",
            "nervous system disorders", "pregnancy, puerperium and perinatal conditions",
            "product issues", "psychiatric disorders",
            "renal and urinary disorders", "reproductive system and breast disorders",
            "respiratory, thoracic and mediastinal disorders",
            "skin and subcutaneous tissue disorders", "social circumstances",
            "surgical and medical procedures", "vascular disorders"
        ]
        if endpoint_lower in sider_endpoints:
            return get_sider_endpoint_description(endpoint)
        
        # elif endpoint_lower == 'herg':
        #     return (
        #         "The following molecule blocks the hERG (human Ether-à-go-go-Related Gene) channel, "
        #         "which is crucial for the coordination of the heart's beating. "
        #         "Blocking the hERG channel can lead to severe adverse effects, including cardiac arrhythmias and sudden cardiac death."
        #     )
        # elif endpoint_lower == 'herg_inhib':
        #     return (
        #         "The following molecule blocks the hERG (human Ether-à-go-go-Related Gene) channel, "
        #         "which is crucial for the coordination of the heart's beating. "
        #         "This dataset evaluates hERG inhibition at multiple concentrations (1uM, 10uM) and can lead to cardiac arrhythmias and sudden cardiac death."
        #     )
        elif endpoint_lower == 'herg_unified':
            return (
                "The following molecule has been evaluated for hERG (human Ether-à-go-go-Related Gene) channel blocking activity "
                "under a unified hERG toxicity endpoint that combines data from the hERG, hERG inhibition, and hERG Karim sources. "
                "Blockade of the hERG channel can disrupt cardiac repolarization and lead to serious adverse effects, "
                "including cardiac arrhythmias and sudden cardiac death."
            )
        elif endpoint_lower == 'ames':
            return (
                "The following molecule is mutagenic, meaning it can cause genetic alterations and DNA damage that may lead to cell death or severe adverse effects."
            )
        elif endpoint_lower == 'clintox':
            return (
                "The following molecule has been associated with clinical toxicity, including drugs that have failed clinical trials due to toxicity reasons."
            )
        elif endpoint_lower == 'dilist':
            return (
                "The following molecule is highly likely to cause human liver injury, or actual cases of liver injury have been reported and confirmed."
            )
        elif endpoint_lower == 'cyp1a2_veith':
            return (
                "The following molecule inhibits CYP P450 1A2 (Veith et al.). "
                "The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. "
                "Specifically, CYP1A2 localizes to the endoplasmic reticulum and its expression can be induced by polycyclic aromatic hydrocarbons (PAHs), "
                "some of which are found in cigarette smoke. It can metabolize PAHs to carcinogenic intermediates and also processes xenobiotics such as "
                "caffeine, aflatoxin B1, and acetaminophen. Inhibition can reduce drug metabolism and increase drug-drug interaction risk."
            )
        elif endpoint_lower == 'cyp2c19_veith':
            return (
                "The following molecule inhibits CYP P450 2C19 (Veith et al.). "
                "The CYP P450 genes are essential for the breakdown (metabolism) of various molecules and chemicals within cells. "
                "Inhibiting these enzymes can lead to poor metabolism of this drug and co-administered drugs, increasing the risk of "
                "drug-drug interactions and adverse effects. CYP2C19 is associated with endoplasmic reticulum functions related to protein processing and transport."
            )
        elif endpoint_lower == 'cyp2c9_veith':
            return (
                "The following molecule inhibits CYP P450 2C9 (Veith et al.). "
                "The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. "
                "Specifically, CYP2C9 plays a major role in oxidation of both xenobiotic and endogenous compounds. "
                "Inhibition can impair metabolic clearance and increase adverse event risk."
            )
        elif endpoint_lower == 'cyp2d6_veith':
            return (
                "The following molecule inhibits CYP P450 2D6 (Veith et al.). "
                "The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. "
                "CYP2D6 is primarily expressed in the liver and is also highly expressed in regions of the central nervous system, including the substantia nigra. "
                "Inhibition can alter metabolic clearance and increase potential toxicity or interaction risk."
            )
        elif endpoint_lower == 'cyp3a4_veith':
            return (
                "The following molecule inhibits CYP P450 3A4 (Veith et al.). "
                "The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. "
                "CYP3A4 is an important enzyme mainly found in the liver and intestine, and oxidizes many foreign organic molecules (xenobiotics), "
                "including toxins and drugs, to support elimination. Inhibition can reduce clearance and increase drug-drug interaction risk."
            )
    
    if not dataset_name:
        return ""
    
    dataset_name_lower = dataset_name.lower()
    
    # Dataset-level descriptions (no endpoint needed)
    if dataset_name_lower == "dilist":
        return (
            "The following molecule is highly likely to cause human liver injury, or actual cases of liver injury have been reported and confirmed."
        )
    elif dataset_name_lower == "dictrank":
        return (
            "The following molecule is highly likely to cause human cardiotoxicity, or actual cases of cardiotoxicity have been reported and confirmed."
        )
    elif dataset_name_lower == "diril":
        return (
            "The following molecule is highly likely to cause human renal toxicity, or actual cases of renal toxicity have been reported and confirmed."
        )
    elif dataset_name_lower == "ames":
        return (
            "The following molecule is mutagenic, meaning it can cause genetic alterations and DNA damage that may lead to cell death or severe adverse effects."
        )
    elif dataset_name_lower == "herg_karim":
        return (
            "The following molecule has been evaluated for hERG (human Ether-à-go-go-Related Gene) channel blocking activity. "
            "This dataset consists of hERG blockers (<10uM) and non-hERG blockers (>=10uM) from integrated sources. "
            "Blocking the hERG channel can lead to cardiac arrhythmias and sudden cardiac death."
        )
    elif dataset_name_lower == "herg" or dataset_name_lower == "herg_inhib":
        return (
            "The following molecule blocks the hERG (human Ether-à-go-go-Related Gene) channel, "
            "which is crucial for the coordination of the heart's beating. "
            "Blocking the hERG channel can lead to severe adverse effects, including cardiac arrhythmias and sudden cardiac death."
        )
    elif dataset_name_lower == "herg_unified":
        return (
            "The following molecule has been evaluated for hERG (human Ether-à-go-go-Related Gene) channel blocking activity "
            "using a unified benchmark that integrates hERG, hERG inhibition, and hERG Karim sources. "
            "Blocking the hERG channel can lead to severe adverse effects, including cardiac arrhythmias and sudden cardiac death."
        )
    elif dataset_name_lower == "skin_reaction":
        return (
            "The following molecule can cause skin sensitization, an immune reaction that leads to allergic contact dermatitis upon repeated exposure."
        )
    elif dataset_name_lower == "carcinogens_lagunin":
        return (
            "The following molecule is carcinogenic, meaning it promotes cancer formation through DNA damage or disruption of cellular metabolic processes."
        )
    elif dataset_name_lower == "clintox":
        return (
            "The following molecule has been associated with clinical toxicity, including drugs that have failed clinical trials due to toxicity reasons."
        )
    
    # Tox21 endpoint-specific contexts
    elif dataset_name_lower == "tox21":
        if endpoint:
            return get_tox21_endpoint_description(endpoint)
        else:
            return (
                "Tox21 Context: Tox21 is a dataset that evaluates chemical compounds across 12 different toxicity endpoints "
                "related to nuclear receptor pathways and stress response pathways.\n"
                "- Toxic: The compound activates or disrupts the specific toxicity pathway being evaluated.\n"
                "- Non-Toxic: The compound does not show significant activity in the evaluated toxicity pathway.\n\n"
            )
    
    # SIDER dataset endpoint-specific contexts
    elif dataset_name_lower in ["sider_train", "sider_test", "sider"]:
        if endpoint:
            return get_sider_endpoint_description(endpoint)
        else:
            return (
                "SIDER Context: SIDER (Side Effect Resource) is a dataset that contains information about "
                "recorded side effects of drugs. The following molecule has been associated with adverse drug reactions.\n\n"
            )
    elif dataset_name_lower == "metabolism":
        if endpoint:
            ep = endpoint.lower()
            if ep == "cyp1a2_veith":
                return (
                    "The following molecule inhibits CYP P450 1A2 (Veith et al.). "
                    "The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. "
                    "Specifically, CYP1A2 localizes to the endoplasmic reticulum and its expression can be induced by polycyclic aromatic hydrocarbons (PAHs), "
                    "some of which are found in cigarette smoke. It can metabolize PAHs to carcinogenic intermediates and also processes xenobiotics such as "
                    "caffeine, aflatoxin B1, and acetaminophen. Inhibition can reduce drug metabolism and increase drug-drug interaction risk."
                )
            if ep == "cyp2c19_veith":
                return (
                    "The following molecule inhibits CYP P450 2C19 (Veith et al.). "
                    "The CYP P450 genes are essential for the breakdown (metabolism) of various molecules and chemicals within cells. "
                    "Inhibiting these enzymes can lead to poor metabolism of this drug and co-administered drugs, increasing the risk of "
                    "drug-drug interactions and adverse effects. CYP2C19 is associated with endoplasmic reticulum functions related to protein processing and transport."
                )
            if ep == "cyp2c9_veith":
                return (
                    "The following molecule inhibits CYP P450 2C9 (Veith et al.). "
                    "The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. "
                    "Specifically, CYP2C9 plays a major role in oxidation of both xenobiotic and endogenous compounds. "
                    "Inhibition can impair metabolic clearance and increase adverse event risk."
                )
            if ep == "cyp2d6_veith":
                return (
                    "The following molecule inhibits CYP P450 2D6 (Veith et al.). "
                    "The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. "
                    "CYP2D6 is primarily expressed in the liver and is also highly expressed in regions of the central nervous system, including the substantia nigra. "
                    "Inhibition can alter metabolic clearance and increase potential toxicity or interaction risk."
                )
            if ep == "cyp3a4_veith":
                return (
                    "The following molecule inhibits CYP P450 3A4 (Veith et al.). "
                    "The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. "
                    "CYP3A4 is an important enzyme mainly found in the liver and intestine, and oxidizes many foreign organic molecules (xenobiotics), "
                    "including toxins and drugs, to support elimination. Inhibition can reduce clearance and increase drug-drug interaction risk."
                )
        return (
            "Metabolism Context: This endpoint evaluates whether the molecule inhibits key CYP450 enzymes involved in drug metabolism. "
            "Inhibition can reduce metabolic clearance, increase exposure, and cause drug-drug interactions or adverse effects."
        )
    
    return ""


def get_tox21_endpoint_description(endpoint: str) -> str:
    """Description of what a Tox21 assay endpoint measures, e.g. 'NR-AR' or 'NR-ER-LBD'."""
    endpoint_contexts: Dict[str, str] = {
        "NR-AR": (
            "The following molecule activates or disrupts the Androgen Receptor (AR) pathway, "
            "which regulates male sexual development and function. "
            "Disruption of this pathway can affect reproductive development and function."
        ),
        "NR-AR-LBD": (
            "The following molecule binds to the Androgen Receptor Ligand Binding Domain (AR-LBD), "
            "affecting androgen signaling pathways. "
            "This assay evaluates more direct binding mechanisms compared to the full receptor activity assay."
        ),
        "NR-AhR": (
            "The following molecule activates the Aryl Hydrocarbon Receptor (AhR) pathway, "
            "which is involved in xenobiotic metabolism and immune responses. "
            "Activation of this receptor can lead to toxic effects such as liver toxicity, carcinogenicity, and immunotoxicity."
        ),
        "NR-Aromatase": (
            "The following molecule inhibits or activates Aromatase, an enzyme essential for estrogen (female hormone) biosynthesis. "
            "This assay evaluates whether the chemical can affect aromatase enzyme activity, thereby influencing estrogen levels. "
            "Disruption of estrogen balance is important for reproductive health."
        ),
        "NR-ER": (
            "The following molecule activates or disrupts the Estrogen Receptor (ER) pathway, "
            "which regulates female sexual development and function. "
            "Disruption of this pathway can affect female reproductive development and function, and is associated with conditions such as breast cancer."
        ),
        "NR-ER-LBD": (
            "The following molecule binds to the Estrogen Receptor Ligand Binding Domain (ER-LBD), "
            "affecting estrogen signaling pathways. "
            "This assay evaluates more direct binding mechanisms compared to the full receptor activity assay."
        ),
        "NR-PPAR-gamma": (
            "The following molecule activates or disrupts the Peroxisome Proliferator-Activated Receptor gamma (PPAR-gamma) pathway, "
            "which regulates glucose and lipid metabolism, cell differentiation, and inflammatory responses. "
            "This assay evaluates whether the chemical can activate PPAR-gamma, potentially affecting metabolic diseases such as diabetes and obesity."
        ),
        "SR-ARE": (
            "The following molecule activates the Antioxidant Response Element (ARE) pathway, "
            "which regulates cellular antioxidant defense mechanisms. "
            "This assay evaluates whether the chemical can activate or inhibit the cell's antioxidant defense system in response to oxidative stress."
        ),
        "SR-ATAD5": (
            "The following molecule affects ATAD5 (ATPase family AAA domain-containing protein 5), "
            "which plays an important role in DNA damage response and repair. "
            "This assay evaluates whether the chemical can affect the ATAD5 pathway, potentially causing DNA damage and genomic instability issues."
        ),
        "SR-HSE": (
            "The following molecule activates the Heat Shock Response Element (HSE) pathway, "
            "which responds to cellular stress and protein misfolding. "
            "Cells respond to protein denaturation stress (heat, toxic substances, etc.) by inducing the production of heat shock proteins (HSP) to repair damaged proteins. "
            "This assay evaluates whether the chemical disrupts the cell's protein quality control system."
        ),
        "SR-MMP": (
            "The following molecule affects Mitochondrial Membrane Potential (MMP), "
            "which is an important indicator of mitochondrial functional status. "
            "Mitochondria are the cell's energy production factories. "
            "This assay evaluates whether the chemical can damage mitochondrial function, potentially causing problems with cellular energy production, which is one of the important mechanisms of cell toxicity."
        ),
        "SR-p53": (
            "The following molecule activates or disrupts the p53 pathway, a critical tumor suppressor pathway involved in cell cycle control and apoptosis. "
            "p53 is known as the 'guardian of the genome' and responds to DNA damage and cellular stress by inducing cell cycle arrest, DNA repair, and apoptosis (cell death). "
            "This assay evaluates whether the chemical can affect the p53 pathway, potentially causing DNA damage, cell death, or cancer development."
        ),
    }
    
    return endpoint_contexts.get(
        endpoint,
        "The following molecule activates or disrupts a specific toxicity pathway being evaluated in the Tox21 dataset."
    )


def get_sider_endpoint_description(endpoint: str) -> str:
    """Description of a SIDER system organ class, e.g. 'Blood and lymphatic system disorders'."""
    endpoint_descriptions: Dict[str, str] = {
        "Blood and lymphatic system disorders": (
            "The following molecule has been associated with blood and lymphatic system disorders, "
            "which can include conditions affecting blood cells, clotting mechanisms, or lymphatic circulation. "
            "These disorders may manifest as anemia, bleeding disorders, or immune system complications."
        ),
        "Cardiac disorders": (
            "The following molecule has been associated with cardiac disorders, "
            "which can include arrhythmias, heart failure, myocardial infarction, or other cardiovascular complications. "
            "These conditions can significantly impact heart function and overall cardiovascular health."
        ),
        "Congenital, familial and genetic disorders": (
            "The following molecule has been associated with congenital, familial, and genetic disorders, "
            "which may involve birth defects, inherited conditions, or genetic mutations. "
            "These disorders can affect development, growth, or long-term health outcomes."
        ),
        "Ear and labyrinth disorders": (
            "The following molecule has been associated with ear and labyrinth disorders, "
            "which can include hearing loss, tinnitus, vertigo, or balance problems. "
            "These conditions can affect auditory function and spatial orientation."
        ),
        "Eye disorders": (
            "The following molecule has been associated with eye disorders, "
            "which can include vision impairment, retinal damage, cataracts, or other ocular complications. "
            "These conditions can significantly impact visual function and quality of life."
        ),
        "General disorders and administration site conditions": (
            "The following molecule has been associated with general disorders and administration site conditions, "
            "which can include injection site reactions, systemic reactions, or general malaise. "
            "These conditions may occur at the site of drug administration or manifest as systemic effects."
        ),
        "Hepatobiliary disorders": (
            "The following molecule has been associated with hepatobiliary disorders, "
            "which can include liver damage, hepatitis, cholestasis, or other liver and bile duct complications. "
            "These conditions can significantly impact liver function and metabolic processes."
        ),
        "Immune system disorders": (
            "The following molecule has been associated with immune system disorders, "
            "which can include autoimmune reactions, hypersensitivity, immunosuppression, or other immune-related complications. "
            "These conditions can affect the body's ability to fight infections or maintain immune homeostasis."
        ),
        "Infections and infestations": (
            "The following molecule has been associated with infections and infestations, "
            "which may indicate increased susceptibility to infections or direct infectious complications. "
            "These conditions can result from immunosuppression or other mechanisms that compromise immune defenses."
        ),
        "Injury, poisoning and procedural complications": (
            "The following molecule has been associated with injury, poisoning, and procedural complications, "
            "which can include accidental overdoses, drug interactions, or complications from medical procedures. "
            "These conditions may result from improper use, dosage errors, or adverse interactions."
        ),
        "Investigations": (
            "The following molecule has been associated with abnormal laboratory findings or investigations, "
            "which can include changes in blood chemistry, liver enzymes, kidney function markers, or other diagnostic parameters. "
            "These findings may indicate underlying organ dysfunction or metabolic disturbances."
        ),
        "Metabolism and nutrition disorders": (
            "The following molecule has been associated with metabolism and nutrition disorders, "
            "which can include diabetes, electrolyte imbalances, metabolic syndrome, or nutritional deficiencies. "
            "These conditions can affect energy metabolism, glucose regulation, or nutrient absorption."
        ),
        "Musculoskeletal and connective tissue disorders": (
            "The following molecule has been associated with musculoskeletal and connective tissue disorders, "
            "which can include muscle weakness, joint pain, bone disorders, or connective tissue damage. "
            "These conditions can affect mobility, strength, and structural integrity of the musculoskeletal system."
        ),
        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)": (
            "The following molecule has been associated with neoplasms (tumors), including benign, malignant, and unspecified growths, "
            "as well as cysts and polyps. These conditions involve abnormal cell growth and may indicate carcinogenic potential or tumor-promoting effects."
        ),
        "Nervous system disorders": (
            "The following molecule has been associated with nervous system disorders, "
            "which can include neurotoxicity, seizures, cognitive impairment, or other neurological complications. "
            "These conditions can affect brain function, peripheral nerves, or overall neurological health."
        ),
        "Pregnancy, puerperium and perinatal conditions": (
            "The following molecule has been associated with pregnancy, puerperium, and perinatal conditions, "
            "which can include complications during pregnancy, childbirth, or the postpartum period. "
            "These conditions can affect maternal health, fetal development, or neonatal outcomes."
        ),
        "Product issues": (
            "The following molecule has been associated with product issues, "
            "which can include quality problems, contamination, or manufacturing defects. "
            "These issues may affect drug safety, efficacy, or stability."
        ),
        "Psychiatric disorders": (
            "The following molecule has been associated with psychiatric disorders, "
            "which can include depression, anxiety, psychosis, mood changes, or other mental health complications. "
            "These conditions can significantly impact cognitive function, emotional well-being, and behavioral patterns."
        ),
        "Renal and urinary disorders": (
            "The following molecule has been associated with renal and urinary disorders, "
            "which can include kidney damage, renal failure, urinary tract complications, or other nephrotoxic effects. "
            "These conditions can significantly impact kidney function and fluid-electrolyte balance."
        ),
        "Reproductive system and breast disorders": (
            "The following molecule has been associated with reproductive system and breast disorders, "
            "which can include hormonal imbalances, fertility issues, reproductive organ complications, or breast-related conditions. "
            "These conditions can affect reproductive health, fertility, or hormonal regulation."
        ),
        "Respiratory, thoracic and mediastinal disorders": (
            "The following molecule has been associated with respiratory, thoracic, and mediastinal disorders, "
            "which can include breathing difficulties, lung damage, respiratory infections, or other pulmonary complications. "
            "These conditions can significantly impact respiratory function and oxygen exchange."
        ),
        "Skin and subcutaneous tissue disorders": (
            "The following molecule has been associated with skin and subcutaneous tissue disorders, "
            "which can include rashes, dermatitis, skin irritation, or other dermatological complications. "
            "These conditions can affect skin integrity, appearance, or protective function."
        ),
        "Social circumstances": (
            "The following molecule has been associated with social circumstances, "
            "which may indicate impacts on social functioning, relationships, or daily activities. "
            "These effects may result from physical or psychological side effects that affect quality of life."
        ),
        "Surgical and medical procedures": (
            "The following molecule has been associated with complications from surgical and medical procedures, "
            "which can include adverse reactions during or after medical interventions. "
            "These complications may result from drug interactions, procedural risks, or patient-specific factors."
        ),
        "Vascular disorders": (
            "The following molecule has been associated with vascular disorders, "
            "which can include blood vessel damage, thrombosis, hypertension, or other circulatory complications. "
            "These conditions can affect blood flow, vascular integrity, or cardiovascular function."
        ),
    }
    
    # Return specific description if available, otherwise return generic SIDER description
    return endpoint_descriptions.get(
        endpoint,
        f"The following molecule has been associated with {endpoint}, which indicates potential adverse effects or toxicity related to this condition."
    )


def get_endpoint_description(dataset_name: Optional[str] = None, endpoint: Optional[str] = None) -> str:
    """Endpoint description for a dataset; thin wrapper over get_dataset_context."""
    return get_dataset_context(dataset_name=dataset_name, endpoint=endpoint)

def _pair_context_for_toxic_nontoxic_tasks() -> str:
    """Context for Task 1, 2, 3: pairs are structurally similar, same endpoint, only toxicity differs."""
    return (
        "Context: The toxic and non-toxic molecules in this task form a pair that is structurally very similar "
        "with minimal physicochemical difference; they differ only in toxicity versus non-toxicity for the same endpoint. "
        "Keep this in mind when performing the task.\n\n"
    )


def _preserve_properties_instruction() -> str:
    """Instruction for Task 2 and 3: preserve other properties, only reduce toxicity for the endpoint."""
    return (
        "When modifying the toxic molecule to make it non-toxic, do not change other physicochemical or "
        "pharmacological properties; only reduce or remove the drug toxicity for this endpoint. "
    )


def _build_safe_explanation() -> str:
    """Return a concise generic SAFE string explanation."""
    return (
        "SAFE (Sequential Attachment-based Fragment Embedding) is a SMILES-compatible string representation "
        "that expresses a molecule as a dot-separated sequence of fragments.\n"
        "\n"
        "How SAFE is constructed:\n"
        "- **Fragmentation**: A molecule is split into fragments by cutting selected bonds using a slicing algorithm.\n"
        "- **Slicer**: The default slicer is `brics`, a rule-based method that cuts retrosynthetically relevant bonds "
        "to produce chemically meaningful substructures.\n"
        "- **Attachment Markers**: At each cut site, attachment information is encoded with SMILES-style ring-closure digits "
        "(e.g., `1`, `2`, ..., `%10`). Matching digits across fragments indicate where fragments reconnect in the full molecule.\n"
        "- **Serialization**: The resulting fragments are written as SMILES strings and joined with `.` separators to form a SAFE string.\n"
        "\n"
        "Important characteristics:\n"
        "- **Fragment-based representation**: Each token block corresponds to a substructure rather than the entire molecule.\n"
        "- **Order invariance**: Changing the fragment order does not change the reconstructed molecule.\n"
        "- **Partial structures**: Individual fragments may look chemically incomplete on their own because they are parts of a larger graph."
    )


def _common_question_preamble() -> str:
    """
    Shared fixed preamble for QA questions (SAFE explanation + pair context + preserve-properties text).
    """
    return (
        _build_safe_explanation()
        + "\n\n"
        + _pair_context_for_toxic_nontoxic_tasks()
        + _preserve_properties_instruction()
        + "\n\n"
    )


def _required_endpoint_block(
    dataset_name: Optional[str],
    endpoint: Optional[str],
    *,
    include_endpoint_description: bool = True,
) -> str:
    """
    Raise if endpoint description cannot be resolved when descriptions are required.
    """
    if not include_endpoint_description:
        return ""
    ds = (dataset_name or "").strip() if isinstance(dataset_name, str) else (dataset_name or "")
    ep = (endpoint or "").strip() if isinstance(endpoint, str) else (endpoint or "")

    # Allow empty dataset/endpoint for ICL demos; cumulative count is printed at build end.
    if not ds and not ep:
        global _MISSING_BOTH_NONE_COUNT
        _MISSING_BOTH_NONE_COUNT += 1
        return ""

    endpoint_desc = get_dataset_context(dataset_name=dataset_name, endpoint=endpoint)
    if not (endpoint_desc or "").strip():
        raise ValueError(
            f"Missing endpoint description for dataset={dataset_name!r}, endpoint={endpoint!r}. "
            "Add the mapping in MolDetox_QA_template.py (bundled endpoint block at file bottom)."
        )
    return endpoint_desc.strip() + "\n\n"


_MISSING_BOTH_NONE_COUNT = 0


def _report_missing_both_none_count() -> None:
    if _MISSING_BOTH_NONE_COUNT > 0:
        print(
            f"[qa_template] skipped endpoint description for dataset=None, endpoint=None: {_MISSING_BOTH_NONE_COUNT} cases",
            file=sys.stderr,
        )


atexit.register(_report_missing_both_none_count)

def _smiles_safe_matching(
    task_name: str,
    toxic_safe: str,
    nontoxic_safe: str,
    toxic_safe_decoded_smiles: str,
    nontoxic_safe_decoded_smiles: str,
    molecule_repr: str = "both_repre",
) -> str:
    """
    Build the molecule representation block for Task 2, Task 1, and Task 3 QA questions.

    molecule_repr: "only_smiles" | "only_safe" | "both_repre"
      - only_smiles: SMILES only
      - only_safe: SAFE only
      - both_repre: SMILES + SAFE and state they denote the same molecule
    """
    t_safe = (toxic_safe or "").strip()
    n_safe = (nontoxic_safe or "").strip()
    t_smiles = (toxic_safe_decoded_smiles or "").strip()
    n_smiles = (nontoxic_safe_decoded_smiles or "").strip()
    repr_type = (molecule_repr or "both_repre").strip().lower()

    def _toxic_line() -> str:
        if repr_type == "only_smiles":
            return f"- Toxic molecule: SMILES = {t_smiles!r}" if t_smiles else ""
        if repr_type == "only_safe":
            return f"- Toxic molecule: SAFE = {t_safe!r}" if t_safe else ""
        # both_repre
        if t_smiles or t_safe:
            return f"- Toxic molecule (same molecule): SMILES = {t_smiles!r}, SAFE = {t_safe!r}"
        return ""

    if task_name in ("task1", "task2", "task3"):
        lt = _toxic_line()
        if not lt:
            return ""
        # strip "- Toxic molecule: " or "- Toxic molecule (same molecule): " prefix
        content = lt.replace("- Toxic molecule (same molecule): ", "").replace("- Toxic molecule: ", "").strip()
        return "Full molecule representation (toxic): " + content + "\n\n"
    return ""


def toxic_molecule_content_for_repr(
    toxic_safe: str,
    toxic_safe_decoded_smiles: str,
    molecule_repr: str = "both_repre",
) -> str:
    """
    One-line toxic molecule caption using the same rules as `_smiles_safe_matching` (task1/2/3).

    Used when ICL few-shot snippets must mirror the molecule representation convention.

    molecule_repr: "only_smiles" | "only_safe" | "both_repre"
    Returns e.g. SMILES = '...', SAFE = '...', or SMILES = '...', SAFE = '...'.
    """
    t_safe = (toxic_safe or "").strip()
    t_smiles = (toxic_safe_decoded_smiles or "").strip()
    repr_type = (molecule_repr or "both_repre").strip().lower()
    if repr_type == "only_smiles":
        return f"SMILES = {t_smiles!r}" if t_smiles else ""
    if repr_type == "only_safe":
        return f"SAFE = {t_safe!r}" if t_safe else ""
    if t_smiles or t_safe:
        return f"SMILES = {t_smiles!r}, SAFE = {t_safe!r}"
    return ""


def task2_nontoxic_fragment_generation(
    toxic_safe: str,
    only_toxic_safe_fragments: str,
    only_nontoxic_safe_fragments: str,
    dataset_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    toxic_safe_decoded_smiles: str = "",
    nontoxic_safe_decoded_smiles: str = "",
    nontoxic_safe: str = "",
    step: str = "multi_step",
    include_output_format: bool = True,
    molecule_repr: str = "both_repre",
    include_endpoint_description: bool = True,
) -> tuple:
    """Task 2: given the toxic molecule and its toxic fragment(s), generate the replacement(s).

    step picks the wording and answer format for one or several fragments.
    include_output_format=False drops the trailing format instruction, for multiple choice.
    """
    endpoint_block = _required_endpoint_block(
        dataset_name=dataset_name,
        endpoint=endpoint,
        include_endpoint_description=include_endpoint_description,
    )
    full_mol_block = _smiles_safe_matching(
        "task2", toxic_safe, nontoxic_safe, toxic_safe_decoded_smiles, nontoxic_safe_decoded_smiles,
        molecule_repr=molecule_repr,
    )

    if step == "single_step":
        task2_output_format = (
            'Output format: a single JSON object with key "answer" and value the single only_nontoxic_safe_fragment string. '
            'Example: {"answer": "frag"}'
        )
        task2_fragment_line = (
            f"- The single fragment that appears only in the toxic molecule (candidate for toxicity-associated structure for this endpoint) is: {only_toxic_safe_fragments}\n\n"
            "Task: Output the only_nontoxic_safe_fragment—i.e. the single SAFE fragment that, when used in place of the only_toxic_safe_fragment, yields a non-toxic molecule for this endpoint. "
            + _preserve_properties_instruction()
        )
    else:
        task2_output_format = (
            'Output format: a single JSON object with key "answer" and value the only_nontoxic_safe_fragments string '
            '(dot-separated for multiple fragments). Example: {"answer": "frag1.frag2"}'
        )
        task2_fragment_line = (
            f"- The fragments that appear only in the toxic molecule (candidates for toxicity-associated structure for this endpoint) are: {only_toxic_safe_fragments}\n\n"
            "Task: Output the only_nontoxic_safe_fragments—i.e. the SAFE fragment(s) that, when used in place of the only_toxic_safe_fragments, yield a non-toxic molecule for this endpoint. "
            + _preserve_properties_instruction()
        )

    task2_question = (
        endpoint_block
        + _common_question_preamble()
        + (full_mol_block if full_mol_block else "")
        + "\n"
        + task2_fragment_line
    ).strip()
    if include_output_format:
        task2_question += " " + task2_output_format

    task2_answer = {"answer": only_nontoxic_safe_fragments or ""}

    return task2_question, task2_answer


def task1_toxic_fragment_identification(
    toxic_safe: str,
    only_toxic_safe_fragments: str,
    dataset_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    toxic_safe_decoded_smiles: str = "",
    step: str = "multi_step",
    include_output_format: bool = True,
    molecule_repr: str = "both_repre",
    include_endpoint_description: bool = True,
) -> tuple:
    """Task 1: given a toxic molecule in SAFE, identify the fragment(s) behind its toxicity.

    step picks the wording and answer format for one or several fragments.
    include_output_format=False drops the trailing format instruction, for multiple choice.
    """

    endpoint_block = _required_endpoint_block(
        dataset_name=dataset_name,
        endpoint=endpoint,
        include_endpoint_description=include_endpoint_description,
    )
    full_mol_block = _smiles_safe_matching(
        "task1", (toxic_safe or "").strip(), "", (toxic_safe_decoded_smiles or "").strip(), "",
        molecule_repr=molecule_repr,
    )

    toxic_safe = (toxic_safe or "").strip()

    if step == "single_step":
        task1_output_format = (
            'Output format: a single JSON object with key "answer" and value the single only_toxic_safe_fragment string. '
            'Example: {"answer": "frag"}'
        )
        task1_instruction = (
            "Task: This toxic molecule belongs to a structurally similar pair that differs only in toxicity for this endpoint. "
            "Identify the single fragment that is the candidate for toxicity-associated structure (the part that drives toxicity for this endpoint) "
            "and output it as only_toxic_safe_fragment. "
        )
    else:
        task1_output_format = (
            'Output format: a single JSON object with key "answer" and value the only_toxic_safe_fragments string '
            '(dot-separated for multiple fragments). Example: {"answer": "frag1.frag2"}'
        )
        task1_instruction = (
            "Task: This toxic molecule belongs to a structurally similar pair that differs only in toxicity for this endpoint. "
            "Identify the fragment(s) that are candidates for toxicity-associated structure (the part(s) that drive toxicity for this endpoint) "
            "and output them as only_toxic_safe_fragments (dot-separated if multiple). "
        )

    task1_question = (
        endpoint_block
        + _common_question_preamble()
        + (full_mol_block if full_mol_block else "")
        + "\n"
        + f"- Toxic molecule (SAFE representation): {toxic_safe}\n\n"
        + task1_instruction
    ).strip()
    if include_output_format:
        task1_question += " " + task1_output_format

    task1_answer = {"answer": (only_toxic_safe_fragments or "").strip()}
    return task1_question, task1_answer


def task3_nontoxic_smiles_generation(
    toxic_safe: str,
    dataset_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    toxic_safe_decoded_smiles: str = "",
    nontoxic_safe_decoded_smiles: str = "",
    step: str = "multi_step",
    include_output_format: bool = True,
    molecule_repr: str = "both_repre",
    include_endpoint_description: bool = True,
) -> tuple:
    """Task 3: nontoxic_smiles_generation — end-to-end.

    LLM receives toxic molecule's SAFE and SMILES; performs identification (Task 1) and
    replacement (Task 2) in one step; outputs nontoxic_safe_decoded_smiles (full SMILES of
    the non-toxic molecule) as the answer.
    """
    endpoint_block = _required_endpoint_block(
        dataset_name=dataset_name,
        endpoint=endpoint,
        include_endpoint_description=include_endpoint_description,
    )
    full_mol_block = _smiles_safe_matching(
        "task3", (toxic_safe or "").strip(), "", (toxic_safe_decoded_smiles or "").strip(), "",
        molecule_repr=molecule_repr,
    )

    if step == "single_step":
        task3_instruction = (
            "Task: From the toxic molecule above, identify the single fragment that is the candidate for "
            "toxicity-associated structure for this endpoint, then determine the single replacement fragment "
            "that yields a non-toxic molecule. Output the resulting non-toxic molecule as a single SMILES string "
            "(nontoxic_safe_decoded_smiles). "
            + _preserve_properties_instruction()
        )
    else:
        task3_instruction = (
            "Task: From the toxic molecule above, identify the fragment(s) that are candidates for "
            "toxicity-associated structure for this endpoint, then determine the replacement fragment(s) "
            "that yield a non-toxic molecule. Output the resulting non-toxic molecule as a single SMILES string "
            "(nontoxic_safe_decoded_smiles). "
            + _preserve_properties_instruction()
        )

    task3_output_format = (
        'Output format: a single JSON object with key "answer" and value the nontoxic_safe_decoded_smiles string. '
        'Example: {"answer": "CCO"}'
    )

    task3_question = (
        endpoint_block
        + _common_question_preamble()
        + (full_mol_block if full_mol_block else "")
        + "\n"
        + task3_instruction
    ).strip()
    if include_output_format:
        task3_question += " " + task3_output_format

    task3_answer = {"answer": (nontoxic_safe_decoded_smiles or "").strip()}
    return task3_question, task3_answer


def task3_stepwise_cot_nontoxic_safe_generation(
    toxic_safe: str,
    dataset_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    toxic_safe_decoded_smiles: str = "",
    nontoxic_safe: str = "",
    # Gold labels for evaluation (NOT shown in the question)
    only_toxic_safe_fragments: str = "",
    only_nontoxic_safe_fragments: str = "",
    step: str = "multi_step",
    include_output_format: bool = True,
    molecule_repr: str = "both_repre",
    include_endpoint_description: bool = True,
) -> tuple:
    """
    Task 3 stepwise CoT variant: Step 1/2 mirror the SAFE-fragment prompts; the final JSON field is **full nontoxic SAFE**.

    Gold answer is nontoxic_safe (full SAFE). Metrics match task3_nontoxic_safe_generation applied to Step 3.
    """
    endpoint_block = _required_endpoint_block(
        dataset_name=dataset_name,
        endpoint=endpoint,
        include_endpoint_description=include_endpoint_description,
    )
    full_mol_block = _smiles_safe_matching(
        "task3",
        (toxic_safe or "").strip(),
        "",
        (toxic_safe_decoded_smiles or "").strip(),
        "",
        molecule_repr=molecule_repr,
    )

    step1_name = "only_toxic_safe_fragments"
    step2_name = "only_nontoxic_safe_fragments"
    if step == "single_step":
        step1_hint = "Identify the single fragment most likely responsible for toxicity for this endpoint."
        step2_hint = (
            "Propose a single non-toxic replacement fragment that reduces toxicity for this endpoint while keeping the overall scaffold as similar as possible."
        )
    else:
        step1_hint = (
            "Identify the fragment(s) most likely responsible for toxicity for this endpoint (dot-separated if multiple)."
        )
        step2_hint = (
            "Propose non-toxic replacement fragment(s) (dot-separated if multiple) that reduce toxicity for this endpoint while keeping the overall scaffold as similar as possible."
        )

    output_format = (
        "Output format: a single JSON object with the following keys:\n"
        f'- "step1_{step1_name}": string (dot-separated SAFE fragment(s))\n'
        '- "step1_reasoning": string\n'
        f'- "step2_{step2_name}": string (dot-separated SAFE fragment(s))\n'
        '- "step2_reasoning": string\n'
        '- "step3_reasoning": string\n'
        '- "answer": string (the final nontoxic **full SAFE string** for the whole molecule)\n'
        'Example: {"step1_only_toxic_safe_fragments":"frag1.frag2","step1_reasoning":"...","step2_only_nontoxic_safe_fragments":"fragA.fragB","step2_reasoning":"...","step3_reasoning":"...","answer":"CCO.[*:1]"}'
    )

    task_block = (
        "Task: Solve the following in ONE call, step by step, using natural-language reasoning.\n"
        "\n"
        "Step 1 (endpoint-aware toxic fragment identification):\n"
        f"- {step1_hint}\n"
        "- In step1_reasoning, identify which fragment is most likely responsible for toxicity for this endpoint and explain *why* the fragment(s) are toxicity-associated for this endpoint, using brief chemical intuition (no need for citations).\n"
        f"- Output the fragment string as step1_{step1_name}.\n"
        "\n"
        "Step 2 (endpoint-aware non-toxic fragment proposal):\n"
        "- Using the Step 1 fragment as the part to be replaced, propose replacement fragment that reduces toxicity for this endpoint while keeping the overall scaffold as similar as possible.\n"
        f"- {step2_hint}\n"
        "- In step2_reasoning, explain the design intent: what property/alert you are trying to reduce for this endpoint and what you preserve while keeping the overall scaffold as similar as possible.\n"
        f"- Output the fragment string as step2_{step2_name}.\n"
        "\n"
        "Step 3 (construct final non-toxic SAFE):\n"
        "- Combine Step 1 and Step 2: conceptually remove the toxic fragment and add the proposed non-toxic fragment that reduces toxicity for this endpoint while keeping the overall scaffold as similar as possible.\n"
        "- In step3_reasoning, describe at a high level how the final molecule changes relative to the toxic molecule.\n"
        '- Output the final non-toxic **full molecule SAFE string** under the key "answer" (not SMILES).\n'
        "\n"
        "Important:\n"
        f"- {_preserve_properties_instruction()}\n"
        "- Your output must be a SINGLE JSON object.\n"
        "- Do not output any text outside the JSON.\n"
        "- The fragment fields must be SAFE fragment strings (dot-separated if multiple)."
    )

    question = (
        endpoint_block
        + _common_question_preamble()
        + (full_mol_block if full_mol_block else "")
        + "\n"
        + task_block
    ).strip()
    if include_output_format:
        question += "\n\n" + output_format

    answer = {
        "answer": (nontoxic_safe or "").strip(),
        "gold_only_toxic_safe_fragments": (only_toxic_safe_fragments or "").strip(),
        "gold_only_nontoxic_safe_fragments": (only_nontoxic_safe_fragments or "").strip(),
    }
    return question, answer


def task3_nontoxic_safe_generation(
    toxic_safe: str,
    nontoxic_safe: str,
    dataset_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    toxic_safe_decoded_smiles: str = "",
    nontoxic_safe_decoded_smiles: str = "",
    step: str = "multi_step",
    include_output_format: bool = True,
    molecule_repr: str = "both_repre",
    include_endpoint_description: bool = True,
) -> tuple:
    """Task 3: nontoxic_safe_generation — end-to-end.

    LLM receives toxic molecule's SAFE and SMILES; performs identification (Task 1) and
    replacement (Task 2) in one step; outputs nontoxic_safe (full SAFE string of
    the non-toxic molecule) as the answer.
    """
    endpoint_block = _required_endpoint_block(
        dataset_name=dataset_name,
        endpoint=endpoint,
        include_endpoint_description=include_endpoint_description,
    )
    full_mol_block = _smiles_safe_matching(
        "task3", (toxic_safe or "").strip(), "", (toxic_safe_decoded_smiles or "").strip(), "",
        molecule_repr=molecule_repr,
    )

    if step == "single_step":
        task3_instruction = (
            "Task: From the toxic molecule above, identify the single fragment that is the candidate for "
            "toxicity-associated structure for this endpoint, then determine the single replacement fragment "
            "that yields a non-toxic molecule. Output the resulting non-toxic molecule as a single SAFE string "
            "as the nontoxic SAFE string. "
            + _preserve_properties_instruction()
        )
    else:
        task3_instruction = (
            "Task: From the toxic molecule above, identify the fragment(s) that are candidates for "
            "toxicity-associated structure for this endpoint, then determine the replacement fragment(s) "
            "that yield a non-toxic molecule. Output the resulting non-toxic molecule as a single SAFE string "
            "as the nontoxic SAFE string. "
            + _preserve_properties_instruction()
        )

    task3_output_format = (
        'Output format: a single JSON object with key "answer" and value the resulting non-toxic SAFE string. '
        'Example: {"answer": "CCO.[*:1]"}'
    )

    task3_question = (
        endpoint_block
        + _common_question_preamble()
        + (full_mol_block if full_mol_block else "")
        + "\n"
        + task3_instruction
    ).strip()
    if include_output_format:
        task3_question += " " + task3_output_format

    task3_answer = {"answer": (nontoxic_safe or "").strip()}
    return task3_question, task3_answer
