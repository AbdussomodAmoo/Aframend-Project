# Tox21 to Filtered AfroDB Toxicity Prediction Pipeline
# Customized for your specific datasets

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import warnings
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="African Phytochemical Toxicity Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)
warnings.filterwarnings('ignore')

# ==============================================================================
# YOUR SPECIFIC DATASET CONFIGURATION
# ==============================================================================

# Tox21 toxicity endpoints (from your header)
TOX21_ENDPOINTS = [
    'NR-AR',  # Androgen Receptor
    'NR-AR-LBD',  # Androgen Receptor Ligand Binding Domain
    'NR-AhR',  # Aryl hydrocarbon Receptor
    'NR-Aromatase',  # Aromatase
    'NR-ER',  # Estrogen Receptor
    'NR-ER-LBD',  # Estrogen Receptor Ligand Binding Domain
    'NR-PPAR-gamma',  # Peroxisome Proliferator-Activated Receptor gamma
    'SR-ARE',  # Antioxidant Response Element
    'SR-ATAD5',  # ATPase Family AAA Domain Containing 5
    'SR-HSE',  # Heat Shock Element
    'SR-MMP',  # Mitochondrial Membrane Potential
    'SR-p53'  # p53 pathway
]

# Pretty names for endpoints (for better understanding)
ENDPOINT_NAMES = {
    'NR-AR': 'Androgen Receptor Disruption',
    'NR-AR-LBD': 'Androgen Receptor Binding',
    'NR-AhR': 'Aryl Hydrocarbon Receptor',
    'NR-Aromatase': 'Aromatase Inhibition',
    'NR-ER': 'Estrogen Receptor Disruption',
    'NR-ER-LBD': 'Estrogen Receptor Binding',
    'NR-PPAR-gamma': 'PPAR-gamma Activation',
    'SR-ARE': 'Antioxidant Response',
    'SR-ATAD5': 'DNA Damage Response',
    'SR-HSE': 'Heat Shock Response',
    'SR-MMP': 'Mitochondrial Toxicity',
    'SR-p53': 'p53 Tumor Suppressor'
}


# ==============================================================================
# FEATURE EXTRACTION
# ==============================================================================

def extract_molecular_features(smiles):
    """
    Extract molecular features that work well for toxicity prediction
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return [0] * 12  # Return zeros for invalid SMILES

    try:
        features = [
            Descriptors.MolWt(mol),  # Molecular weight
            Descriptors.MolLogP(mol),  # Lipophilicity
            Descriptors.TPSA(mol),  # Topological polar surface area
            Descriptors.NumHDonors(mol),  # Hydrogen bond donors
            Descriptors.NumHAcceptors(mol),  # Hydrogen bond acceptors
            Descriptors.NumRotatableBonds(mol),  # Rotatable bonds
            Descriptors.HeavyAtomCount(mol),  # Heavy atom count
            Descriptors.NumAromaticRings(mol),  # Aromatic rings
            Descriptors.FractionCSP3(mol),  # Fraction sp3 carbons
            Descriptors.BertzCT(mol),  # Molecular complexity
            Descriptors.NumSaturatedRings(mol),  # Saturated rings
            Descriptors.NumAliphaticRings(mol)  # Aliphatic rings
        ]
        return features
    except:
        return [0] * 12


def create_features_dataframe(smiles_list):
    """
    Create feature DataFrame from SMILES list
    """
    feature_names = [
        'molecular_weight', 'logp', 'tpsa', 'hbd', 'hba', 'rotatable_bonds',
        'heavy_atoms', 'aromatic_rings', 'fraction_sp3', 'complexity',
        'saturated_rings', 'aliphatic_rings'
    ]

    print(f"🧬 Extracting features for {len(smiles_list)} molecules...")

    features_list = []
    failed_count = 0

    for i, smiles in enumerate(smiles_list):
        features = extract_molecular_features(smiles)
        features_list.append(features)

        if features == [0] * 12:
            failed_count += 1

        if (i + 1) % 1000 == 0:
            print(f"   Processed {i + 1}/{len(smiles_list)} molecules...")

    print(f"✅ Feature extraction complete! Failed: {failed_count}/{len(smiles_list)}")

    return pd.DataFrame(features_list, columns=feature_names)


# ==============================================================================
# MAIN CLASSES: PLANT SEARCH AND FILTERING AGENTS
# ==============================================================================

class PlantAgent:
    def __init__(self, df):
        self.df = df

    def search_by_plant(self, plant_name, top_n=10):
        """Search for compounds by plant name"""
        results = self.df[self.df['organisms'].str.contains(plant_name, case=False, na=False)]
        if results.empty:
            print(f"❌ No compounds found for: {plant_name}")
            return None
        return results[[
            'identifier', 'name', 'organisms',
            'canonical_smiles', 'molecular_formula',
            'molecular_weight', 'exact_molecular_weight', 'alogp',
            'topological_polar_surface_area', 'rotatable_bond_count',
            'hydrogen_bond_acceptors', 'hydrogen_bond_donors',
            'aromatic_rings_count', 'formal_charge', 'fractioncsp3',
            'qed_drug_likeliness', 'np_likeness', 'contains_sugar',
            'contains_ring_sugars', 'contains_linear_sugars',
            'chemical_class', 'chemical_sub_class', 'chemical_super_class',
            'np_classifier_class', 'np_classifier_superclass', 'np_classifier_pathway'
        ]].head(top_n)

    def list_all_organisms_tabular(self):
        """Returns organisms with occurrence counts in tabular format"""
        from collections import Counter
        organism_series = self.df['organisms'].dropna()
        all_organisms = []

        for entry in organism_series:
            parts = [o.strip() for o in entry.split('|') if o.strip()]
            all_organisms.extend(parts)

        organism_counts = Counter(all_organisms)
        return pd.DataFrame(organism_counts.items(), columns=['Organism', 'Count']).sort_values(by='Count',
                                                                                                ascending=False)

    def suggest_plants(self, partial_name="", top_n=20):
        """Suggest plant names based on partial input"""
        organism_df = self.list_all_organisms_tabular()
        if partial_name:
            suggestions = organism_df[organism_df['Organism'].str.contains(partial_name, case=False, na=False)]
        else:
            suggestions = organism_df

        return suggestions.head(top_n)

class FilterAgent:
    def __init__(self, df):
        self.df = df.copy()

    def lipinski_filter(self, row):
        """Apply Lipinski's rule of five"""
        try:
            mw = row['molecular_weight']
            logp = row['alogp']
            hba = row['hydrogen_bond_acceptors']
            hbd = row['hydrogen_bond_donors']
            rules_passed = sum([
                mw < 500,
                logp < 5,
                hba <= 10,
                hbd <= 5
            ])
            return rules_passed >= 2  # relaxed rule
        except:
            return False

    def apply_filters(self, qed_threshold=0.5, np_threshold=0.3):
        """Apply drug-likeness filters"""
        # Apply Lipinski
        self.df['lipinski_pass'] = self.df.apply(self.lipinski_filter, axis=1)

        # Apply QED filter
        self.df['qed_pass'] = self.df['qed_drug_likeliness'] >= qed_threshold

        # Apply NP-likeness filter
        self.df['np_pass'] = self.df['np_likeness'] >= np_threshold

        # Return filtered DataFrame
        filtered_df = self.df[
            self.df['lipinski_pass'] &
            self.df['qed_pass'] &
            self.df['np_pass']
            ]
        print(f"✅ {len(filtered_df)} compounds passed filters out of {len(self.df)}")
        return filtered_df

# ==============================================================================
# TOXICITY PREDICTION MODEL
# ==============================================================================

class AfrcanPhytochemicalToxicityPredictor:
    """
    Toxicity predictor specifically for African phytochemicals
    """

    def __init__(self):
        self.models = {}
        self.feature_names = []
        self.training_stats = {}

    def train_models(self, tox21_df):
        """
        Train toxicity models using Tox21 data
        """
        print("🤖 Training toxicity prediction models...")
        print(f"   Training on {len(tox21_df)} Tox21 compounds")

        # Extract features from Tox21 SMILES
        print("   Extracting features from Tox21 dataset...")
        tox21_features = create_features_dataframe(tox21_df['smiles'].tolist())
        self.feature_names = tox21_features.columns.tolist()

        progress_bar = st.progress(0)
        status_text = st.empty()
        # Train model for each toxicity endpoint
        for i, endpoint in enumerate(TOX21_ENDPOINTS):
            print(f"   Training model for {ENDPOINT_NAMES[endpoint]}...")

            # Get valid data (remove NaN values)
            valid_mask = ~tox21_df[endpoint].isna()
            X_train = tox21_features[valid_mask]
            y_train = tox21_df[endpoint][valid_mask]

            if len(y_train) == 0:
                print(f"   ⚠️ No valid data for {endpoint}")
                continue

            # Train Random Forest model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced',  # Handle imbalanced data
                n_jobs=-1
            )

            model.fit(X_train, y_train)
            self.models[endpoint] = model

            # Store training statistics
            self.training_stats[endpoint] = {
                'n_samples': len(y_train),
                'n_positive': sum(y_train),
                'n_negative': len(y_train) - sum(y_train),
                'positive_rate': sum(y_train) / len(y_train)
            }

            print(f"   ✅ {endpoint}: {len(y_train)} samples, {sum(y_train)} positive cases")
            progress_bar.progress((i + 1) / len(TOX21_ENDPOINTS))

        print(f"🎯 Training complete! Trained {len(self.models)} models")
        status_text.text("✅ Training complete!")
        return len(self.models)

    def predict_afrodb_toxicity(self, filtered_afrodb_df):
        """
        Predict toxicity for filtered AfroDB compounds
        """
        print("🔮 Making toxicity predictions for African phytochemicals...")

        # Extract features from AfroDB SMILES
        afrodb_features = create_features_dataframe(filtered_afrodb_df['canonical_smiles'].tolist())

        # Make predictions for each endpoint
        predictions = {}

        for endpoint in self.models:
            print(f"   Predicting {ENDPOINT_NAMES[endpoint]}...")

            model = self.models[endpoint]

            # Get probability predictions
            pred_proba = model.predict_proba(afrodb_features)

            # Handle cases where model only learned one class
            if pred_proba.shape[1] == 1:
                toxic_prob = pred_proba[:, 0]
            else:
                toxic_prob = pred_proba[:, 1]  # Probability of toxic class

            predictions[f"{endpoint}_probability"] = toxic_prob
            predictions[f"{endpoint}_prediction"] = ['Toxic' if p > 0.5 else 'Non-toxic' for p in toxic_prob]
            predictions[f"{endpoint}_risk_level"] = [
                'High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' for p in toxic_prob
            ]

        return pd.DataFrame(predictions)

    def predict_single_compound(self, smiles):
        """
        Predict toxicity for a single compound (for LLM integration)
        """
        features = extract_molecular_features(smiles)
        features_array = np.array(features).reshape(1, -1)

        results = {'smiles': smiles, 'predictions': {}}

        for endpoint in self.models:
            model = self.models[endpoint]
            pred_proba = model.predict_proba(features_array)[0]

            toxic_prob = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]

            results['predictions'][endpoint] = {
                'endpoint_name': ENDPOINT_NAMES[endpoint],
                'toxic_probability': round(toxic_prob, 3),
                'prediction': 'Toxic' if toxic_prob > 0.5 else 'Non-toxic',
                'risk_level': 'High' if toxic_prob > 0.7 else 'Medium' if toxic_prob > 0.3 else 'Low'
            }

        # Overall risk assessment
        high_risk_count = sum(1 for p in results['predictions'].values() if p['risk_level'] == 'High')
        results['overall_risk'] = 'High' if high_risk_count >= 2 else 'Medium' if high_risk_count >= 1 else 'Low'

        return results

    def save_models(self, filename='african_phytochemical_toxicity_models.pkl'):
        """
        Save trained models
        """
        model_data = {
            'models': self.models,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'endpoints': TOX21_ENDPOINTS,
            'endpoint_names': ENDPOINT_NAMES
        }
        joblib.dump(model_data, filename)
        print(f"💾 Models saved to {filename}")

    def load_models(self, filename='african_phytochemical_toxicity_models.pkl'):
        """
        Load pre-trained models
        """
        model_data = joblib.load(filename)
        self.models = model_data['models']
        self.feature_names = model_data['feature_names']
        self.training_stats = model_data['training_stats']
        print(f"📂 Models loaded from {filename}")

'''
# ==============================================================================
# MAIN PIPELINE FUNCTION
# ==============================================================================

def run_your_pipeline(tox21_file, filtered_afrodb_df):
    """
    Run the complete pipeline with your specific datasets
    """
    print("🚀 African Phytochemical Toxicity Prediction Pipeline")
    print("=" * 60)

    # Load Tox21 dataset
    print("📁 Loading Tox21 dataset...")
    tox21_df = pd.read_csv(tox21_file)
    print(f"   Loaded {len(tox21_df)} Tox21 compounds")

    # Display dataset info
    print(f"   AfroDB filtered dataset: {len(filtered_afrodb_df)} compounds")

    # Initialize predictor
    predictor = AfrcanPhytochemicalToxicityPredictor()

    # Train models
    predictor.train_models(tox21_df)

    # Save models
    predictor.save_models()

    # Make predictions on filtered AfroDB
    print("🔮 Making predictions on filtered African phytochemicals...")
    toxicity_predictions = predictor.predict_afrodb_toxicity(filtered_afrodb_df)

    # Combine results
    final_results = pd.concat([filtered_afrodb_df.reset_index(drop=True), toxicity_predictions], axis=1)

    # Save results
    output_file = 'african_phytochemicals_toxicity_predictions.csv'
    final_results.to_csv(output_file, index=False)

    print(f"✅ Pipeline complete!")
    print(f"📊 Results saved to: {output_file}")
    print(f"🔧 Models saved for LLM integration")

    # Show summary statistics
    print("\n📈 Prediction Summary:")
    for endpoint in TOX21_ENDPOINTS:
        if f"{endpoint}_risk_level" in final_results.columns:
            risk_counts = final_results[f"{endpoint}_risk_level"].value_counts()
            print(f"   {ENDPOINT_NAMES[endpoint]}:")
            print(f"      High Risk: {risk_counts.get('High', 0)}")
            print(f"      Medium Risk: {risk_counts.get('Medium', 0)}")
            print(f"      Low Risk: {risk_counts.get('Low', 0)}")

    return final_results, predictor




# ==============================================================================
# INTEGRATED PIPELINE FOR STREAMLIT
# ==============================================================================

def load_coconut_data(coconut_file_path):
    """Load COCONUT database"""
    try:
        df = pd.read_csv(coconut_file_path)
        print(f"📊 Loaded {len(df)} compounds from COCONUT database")
        return df
    except FileNotFoundError:
        print(f"❌ Could not find COCONUT database at: {coconut_file_path}")
        return None


def run_plant_toxicity_analysis(coconut_df, tox21_file, plant_name):
    """Complete pipeline for plant-specific toxicity analysis"""
    print(f"🌿 Running toxicity analysis for: {plant_name}")
    print("=" * 60)

    # Step 1: Search for plant compounds
    plant_agent = PlantAgent(coconut_df)
    plant_compounds = plant_agent.search_by_plant(plant_name)

    if plant_compounds is None:
        return None, None, None

    print(f"🔍 Found {len(plant_compounds)} compounds for {plant_name}")

    # Step 2: Apply filters
    filter_agent = FilterAgent(plant_compounds)
    filtered_compounds = filter_agent.apply_filters()

    if len(filtered_compounds) == 0:
        print("❌ No compounds passed the filters")
        return plant_compounds, None, None

    # Step 3: Run toxicity prediction
    results, predictor = run_your_pipeline(tox21_file, filtered_compounds)

    return plant_compounds, filtered_compounds, results


# ==============================================================================
# LLM INTEGRATION FUNCTIONS
# ==============================================================================

def quick_phytochemical_toxicity_check(smiles):
    """
    Quick toxicity check for LLM integration
    """
    predictor = AfrcanPhytochemicalToxicityPredictor()
    predictor.load_models()

    return predictor.predict_single_compound(smiles)


def interpret_toxicity_results(results):
    """
    Create human-readable toxicity report
    """
    report = f"🧪 Toxicity Analysis for: {results['smiles']}\n"
    report += f"🎯 Overall Risk Level: {results['overall_risk']}\n\n"

    report += "📋 Individual Endpoint Analysis:\n"
    for endpoint, data in results['predictions'].items():
        risk_emoji = "🔴" if data['risk_level'] == 'High' else "🟡" if data['risk_level'] == 'Medium' else "🟢"
        report += f"   {risk_emoji} {data['endpoint_name']}: {data['prediction']} ({data['toxic_probability'] * 100:.1f}%)\n"

    return report



# ==============================================================================
# USAGE
# ==============================================================================


if __name__ == "__main__":
    print("🌿 African Phytochemical Toxicity Prediction System")
    print("=" * 50)

    # Usage for local environment
    print("\n🔧 To run with your datasets:")
    print("# Load COCONUT database")
    coconut_df = load_coconut_data("coconut_csv/coconut_csv-06-2025.csv")
    print(plant_compounds, filtered_compounds, results = run_plant_toxicity_analysis(coconut_df, 'tox21.csv', 'Vernonia amygdalina'))

    #print("\n🔍 For single compound analysis:")
    result = quick_phytochemical_toxicity_check('COC1=CC=CC2=C1C(=O)C1=C(O)C3=C...')
    report = interpret_toxicity_results(result)
    print(report)

'''


# Utility functions
def create_toxicity_visualization(results):
    """Create visualization for toxicity results"""
    endpoints = []
    probabilities = []
    risk_levels = []

    for endpoint, data in results['predictions'].items():
        endpoints.append(data['endpoint_name'])
        probabilities.append(data['toxic_probability'])
        risk_levels.append(data['risk_level'])

    # Create color mapping for risk levels
    color_map = {'Low': '#2E8B57', 'Medium': '#FFD700', 'High': '#DC143C'}
    colors = [color_map[level] for level in risk_levels]

    fig = go.Figure(data=[
        go.Bar(
            x=endpoints,
            y=probabilities,
            marker_color=colors,
            text=[f"{p:.1%}" for p in probabilities],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Toxicity Prediction Results",
        xaxis_title="Toxicity Endpoints",
        yaxis_title="Toxic Probability",
        xaxis={'tickangle': 45},
        height=500
    )

    return fig


def main():
    st.title("🌿 African Phytochemical Toxicity Predictor")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Home", "Plant Search", "Toxicity Prediction", "Model Training", "About"]
    )

    if page == "Home":
        st.header("Welcome to the African Phytochemical Toxicity Predictor")
        st.markdown("""
        This application helps predict the toxicity of African phytochemicals using machine learning models 
        trained on the Tox21 dataset. The system can:

        - 🔍 Search for compounds in African plants
        - 🧪 Predict toxicity across multiple endpoints
        - 📊 Visualize results with interactive charts
        - 🎯 Apply drug-likeness filters

        **Get started by selecting a page from the sidebar!**
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Toxicity Endpoints", "12")
        with col2:
            st.metric("Plant Species", "1000+")
        with col3:
            st.metric("Compounds", "10,000+")

    elif page == "Plant Search":
        st.header("🔍 Plant Compound Search")

        # File upload for COCONUT dataset
        uploaded_file = st.file_uploader("Upload COCONUT CSV file", type=['csv'])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} compounds from COCONUT dataset")

                # Initialize plant agent
                plant_agent = PlantAgent(df)

                # Search interface
                plant_name = st.text_input("Enter plant name (e.g., Vernonia amygdalina):")

                if plant_name:
                    results = plant_agent.search_by_plant(plant_name)

                    if results is not None:
                        st.success(f"Found {len(results)} compounds for {plant_name}")
                        st.dataframe(results)

                        # Download button
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name=f"{plant_name}_compounds.csv",
                            mime="text/csv"
                        )

                        # Full toxicity analysis option
                        if st.button("🧪 Run Full Toxicity Analysis"):
                            predictor = AfricanPhytochemicalToxicityPredictor()
                            if predictor.load_models():
                                with st.spinner("Applying filters and making predictions..."):
                                    # Apply filters
                                    filter_agent = FilterAgent(results)
                                    filtered_compounds = filter_agent.apply_filters()

                                    if len(filtered_compounds) > 0:
                                        # Make batch predictions
                                        toxicity_predictions = predictor.predict_afrodb_toxicity(filtered_compounds)

                                        # Combine results
                                        final_results = pd.concat(
                                            [filtered_compounds.reset_index(drop=True), toxicity_predictions], axis=1)

                                        st.success(f"✅ Toxicity analysis complete for {len(final_results)} compounds!")
                                        st.dataframe(final_results)

                                        # Summary statistics
                                        st.subheader("📈 Risk Summary")
                                        for endpoint in TOX21_ENDPOINTS:
                                            if f"{endpoint}_risk_level" in final_results.columns:
                                                risk_counts = final_results[f"{endpoint}_risk_level"].value_counts()
                                                st.write(f"**{ENDPOINT_NAMES[endpoint]}:**")
                                                col1, col2, col3 = st.columns(3)
                                                col1.metric("High Risk", risk_counts.get('High', 0))
                                                col2.metric("Medium Risk", risk_counts.get('Medium', 0))
                                                col3.metric("Low Risk", risk_counts.get('Low', 0))

                                        # Download full results
                                        csv_full = final_results.to_csv(index=False)
                                        st.download_button(
                                            label="Download Full Analysis as CSV",
                                            data=csv_full,
                                            file_name=f"{plant_name}_toxicity_analysis.csv",
                                            mime="text/csv"
                                        )
                                    else:
                                        st.warning("No compounds passed the drug-likeness filters.")
                            else:
                                st.error("No trained models found. Please train models first.")
                    else:
                        st.warning(f"No compounds found for {plant_name}")

                # Show organism statistics
                if st.checkbox("Show organism statistics"):
                    with st.spinner("Generating organism statistics..."):
                        organism_stats = plant_agent.list_all_organisms_tabular()
                        st.subheader("Top Organisms in Dataset")
                        st.dataframe(organism_stats.head(20))

            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
        else:
            st.info("Please upload the COCONUT CSV file to start searching.")

    elif page == "Toxicity Prediction":
        st.header("🧪 Toxicity Prediction")

        # Check if models exist
        predictor = AfricanPhytochemicalToxicityPredictor()
        models_loaded = predictor.load_models()

        if not models_loaded:
            st.warning("⚠️ No trained models found. Please train models first in the 'Model Training' page.")
            return

        st.success("✅ Models loaded successfully!")

        # Input methods
        input_method = st.radio("Choose input method:", ["Enter SMILES", "Upload CSV"])

        if input_method == "Enter SMILES":
            smiles_input = st.text_input("Enter SMILES string:")

            if smiles_input and st.button("Predict Toxicity"):
                with st.spinner("Making predictions..."):
                    try:
                        results = predictor.predict_single_compound(smiles_input)

                        # Display overall risk
                        risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                        st.subheader(f"Overall Risk: {risk_color[results['overall_risk']]} {results['overall_risk']}")

                        # Create visualization
                        fig = create_toxicity_visualization(results)
                        st.plotly_chart(fig, use_container_width=True)

                        # Detailed results table
                        st.subheader("Detailed Results")
                        results_df = pd.DataFrame([
                            {
                                'Endpoint': data['endpoint_name'],
                                'Prediction': data['prediction'],
                                'Probability': f"{data['toxic_probability']:.1%}",
                                'Risk Level': data['risk_level']
                            }
                            for data in results['predictions'].values()
                        ])
                        st.dataframe(results_df)

                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")

        elif input_method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV with SMILES column", type=['csv'])

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    smiles_col = st.selectbox("Select SMILES column:", df.columns)

                    if st.button("Predict Toxicity for All Compounds"):
                        progress_bar = st.progress(0)
                        results_list = []

                        for i, smiles in enumerate(df[smiles_col]):
                            try:
                                result = predictor.predict_single_compound(smiles)
                                results_list.append({
                                    'SMILES': smiles,
                                    'Overall_Risk': result['overall_risk'],
                                    **{f"{ep}_{key}": val for ep, data in result['predictions'].items()
                                       for key, val in data.items()}
                                })
                            except:
                                results_list.append({'SMILES': smiles, 'Overall_Risk': 'Error'})

                            progress_bar.progress((i + 1) / len(df))

                        results_df = pd.DataFrame(results_list)
                        st.success("✅ Predictions complete!")
                        st.dataframe(results_df)

                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="toxicity_predictions.csv",
                            mime="text/csv"
                        )

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    elif page == "Model Training":
        st.header("🤖 Model Training")

        # Upload Tox21 dataset
        tox21_file = st.file_uploader("Upload Tox21 CSV file", type=['csv'])

        if tox21_file is not None:
            try:
                tox21_df = pd.read_csv(tox21_file)
                st.success(f"✅ Loaded Tox21 dataset with {len(tox21_df)} compounds")

                # Show dataset info
                st.subheader("Dataset Information")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Total Compounds", len(tox21_df))
                    st.metric("Features", len(tox21_df.columns))

                with col2:
                    endpoint_counts = []
                    for endpoint in TOX21_ENDPOINTS:
                        if endpoint in tox21_df.columns:
                            valid_count = tox21_df[endpoint].notna().sum()
                            endpoint_counts.append(valid_count)

                    if endpoint_counts:
                        st.metric("Avg. Endpoint Data", f"{np.mean(endpoint_counts):.0f}")
                        st.metric("Endpoints Available", len(endpoint_counts))

                # Training button
                if st.button("🚀 Train Models"):
                    with st.spinner("Training models..."):
                        predictor = AfricanPhytochemicalToxicityPredictor()
                        models_trained = predictor.train_models(tox21_df)

                        if models_trained > 0:
                            # Save models
                            filename = predictor.save_models()
                            st.success(f"✅ Successfully trained {models_trained} models!")
                            st.info(f"Models saved as: {filename}")

                            # Show training statistics
                            st.subheader("Training Statistics")
                            stats_df = pd.DataFrame([
                                {
                                    'Endpoint': ENDPOINT_NAMES[endpoint],
                                    'Total Samples': stats['n_samples'],
                                    'Positive Cases': stats['n_positive'],
                                    'Negative Cases': stats['n_negative'],
                                    'Positive Rate': f"{stats['positive_rate']:.1%}"
                                }
                                for endpoint, stats in predictor.training_stats.items()
                            ])
                            st.dataframe(stats_df)
                        else:
                            st.error("❌ No models were trained. Check your dataset.")

            except Exception as e:
                st.error(f"Error loading Tox21 dataset: {str(e)}")
        else:
            st.info("Please upload the Tox21 CSV file to start training.")

    elif page == "About":
        st.header("ℹ️ About")
        st.markdown("""
        ### African Phytochemical Toxicity Predictor

        This application is designed to predict the toxicity of compounds found in African plants using 
        machine learning models trained on the Tox21 dataset.

        **Key Features:**
        - **Plant Search**: Search for compounds in specific African plants
        - **Toxicity Prediction**: Predict toxicity across 12 different endpoints
        - **Model Training**: Train custom models using your own Tox21 data
        - **Visualization**: Interactive charts and graphs for better understanding

        **Toxicity Endpoints:**
        """)

        for endpoint, name in ENDPOINT_NAMES.items():
            st.write(f"- **{endpoint}**: {name}")

        st.markdown("""
        **Technical Details:**
        - Machine Learning: Random Forest Classifier
        - Feature Extraction: RDKit molecular descriptors
        - Dataset: Tox21 challenge dataset
        - Filtering: Lipinski's Rule of Five, QED, NP-likeness

        **Disclaimer:**
        This tool is for research purposes only. Always consult with experts before making 
        decisions based on these predictions.
        """)


if __name__ == "__main__":
    main()


