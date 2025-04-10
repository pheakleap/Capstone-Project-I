import gradio as gr
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from typing import List, Tuple, Optional, Dict, Set, Any
import networkx as nx
import traceback 
import itertools
# --- Configuration ---
MODEL_PATH = 'models/ensemble_models.joblib'
DESCRIPTION_PATH = 'assets/raw/symptom_Description.csv'
PRECAUTION_PATH = 'assets/raw/symptom_precaution.csv'
SEVERITY_PATH = 'assets/raw/Symptom-severity.csv'
SYMPTOM_ANALYSIS_PATH = 'assets/processed/symptom_analysis.joblib'

# --- Load Data and Models ---
def load_data() -> Tuple[object, List[str], List[str], Set[str], Dict[str, str], List[str], pd.DataFrame, pd.DataFrame, Dict[str, int], Dict]:
    """Loads all necessary data files and the model, returning cleaned data structures."""
    try:
        results = joblib.load(MODEL_PATH)
        voting_clf = results['voting_classifier']
        # Store raw feature names exactly as the model expects them
        raw_feature_names: List[str] = results['feature_names']
        target_names: List[str] = results['target_names']

        description_df = pd.read_csv(DESCRIPTION_PATH)
        precaution_df = pd.read_csv(PRECAUTION_PATH)
        symptom_severity_df = pd.read_csv(SEVERITY_PATH)

        # Load correlation data carefully
        symptom_analysis = {'correlation_matrix': pd.DataFrame()} # Default empty
        try:
            loaded_analysis = joblib.load(SYMPTOM_ANALYSIS_PATH)
            if 'correlation_matrix' in loaded_analysis and isinstance(loaded_analysis['correlation_matrix'], pd.DataFrame):
                symptom_analysis = loaded_analysis
            else:
                print("Warning: 'correlation_matrix' not found or invalid in symptom_analysis.joblib.")
        except Exception as e:
            print(f"Warning: Could not load or parse {SYMPTOM_ANALYSIS_PATH}: {e}. Network graph may not work.")

        # Preprocess severity data (keys are underscore format)
        symptom_severity_df['Symptom'] = symptom_severity_df['Symptom'].str.strip().str.lower().str.replace(' ', '_')
        severity_info = dict(zip(symptom_severity_df['Symptom'], symptom_severity_df['weight']))

        # **** Crucial Cleaning Step ****
        # Create a cleaned list and set for UI and internal logic
        # Also create a mapping from cleaned name back to raw name for model input
        cleaned_feature_names_list = []
        cleaned_to_raw_map = {}
        for name in raw_feature_names:
            cleaned_name = name.strip() # Basic cleaning
            if cleaned_name not in cleaned_to_raw_map: # Avoid duplicates if raw names had variations
                 cleaned_feature_names_list.append(cleaned_name)
                 cleaned_to_raw_map[cleaned_name] = name # Map cleaned back to the first raw version encountered

        cleaned_feature_names_list = sorted(cleaned_feature_names_list)
        cleaned_feature_names_set = set(cleaned_feature_names_list)

        print(f"Loaded {len(raw_feature_names)} raw feature names.")
        print(f"Created {len(cleaned_feature_names_set)} unique cleaned feature names for UI.")

        return (voting_clf, raw_feature_names, cleaned_feature_names_list, cleaned_feature_names_set,
                cleaned_to_raw_map, target_names, description_df, precaution_df, severity_info, symptom_analysis)

    except FileNotFoundError as e:
        print(f"FATAL Error loading essential data: {e}. Check paths relative to 'app/'. Exiting.")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        traceback.print_exc()
        raise e

try:
    (voting_clf, raw_feature_names, cleaned_feature_names, feature_names_set, cleaned_to_raw_map,
     target_names, description_df, precaution_df, severity_info, symptom_analysis) = load_data()
except Exception:
    exit()

# --- Symptom Categorization & Preparation ---
# Use raw categories dict, but filter based on *cleaned* feature names set
symptom_categories_raw = {
    "General": [" fatigue", " lethargy", " malaise", " weakness_in_limbs", " weight_loss", " weight_gain", " high_fever", " mild_fever", " sweating", " chills", " dehydration", " shivering"],
    "Pain/Discomfort": [" headache", " stomach_pain", " abdominal_pain", " back_pain", " chest_pain", " joint_pain", " muscle_pain", " neck_pain", " knee_pain", " hip_joint_pain", " stiff_neck", " muscle_wasting", " cramps", " pain_behind_the_eyes"],
    "Skin/External": ["itching", " skin_rash", " nodal_skin_eruptions", " yellowish_skin", " bruising", " red_spots_over_body", " dischromic _patches", " toxic_look_(typhos)", " yellow_crust_ooze", " pus_filled_pimples", " blackheads", " scurring", " skin_peeling", " blister", " red_sore_around_nose"],
    "Respiratory": [" cough", " breathlessness", " runny_nose", " congestion", " sinus_pressure", " phlegm", " throat_irritation", " continuous_sneezing", " mucoid_sputum", " rusty_sputum", " blood_in_sputum"],
    "Digestive": [" nausea", " vomiting", " diarrhoea", " constipation", " stomach_bleeding", " distention_of_abdomen", " acidity", " ulcers_on_tongue", " loss_of_appetite", " excessive_hunger", " indigestion", " passage_of_gases", " internal_itching", " bloody_stool"],
    "Neurological": [" dizziness", " unsteadiness", " lack_of_concentration", " altered_sensorium", " depression", " irritability", " slurred_speech", " visual_disturbances", " anxiety", " loss_of_balance", " loss_of_smell", " mood_swings", " coma"],
    "Cardiovascular": [" chest_pain", " palpitations", " fast_heart_rate", " swollen_blood_vessels", " prominent_veins_on_calf", " cold_hands_and_feets", " fluid_overload"],
    "Urinary/Diabetes": [" burning_micturition", " spotting_ urination", " dark_urine", " yellow_urine", " polyuria", " bladder_discomfort", " continuous_feel_of_urine", " irregular_sugar_level", " increased_appetite", " foul_smell_of urine"],
    "Eyes/Face": [" puffy_face_and_eyes", " sunken_eyes", " redness_of_eyes", " watering_from_eyes", " yellowing_of_eyes", " blurred_and_distorted_vision"],
    "Musculoskeletal": [" muscle_weakness", " movement_stiffness", " swelling_joints", " painful_walking", " swollen_legs", " swollen_extremeties", " swelling_of_stomach"],
    "Medical History": [" family_history", " history_of_alcohol_consumption", " extra_marital_contacts", " receiving_blood_transfusion", " receiving_unsterile_injections"],
    "Others": []
}

original_choices = {} # Stores {category_name: [list_of_cleaned_symptoms]}
all_categorized_symptoms = set()

for category, symptoms in symptom_categories_raw.items():
    cleaned_symptoms_in_category = [s.strip() for s in symptoms]
    # Use the cleaned feature names set for filtering
    valid_symptoms = sorted([s for s in cleaned_symptoms_in_category if s in feature_names_set])
    if valid_symptoms:
        clean_category_name = category.replace("/", " or ")
        original_choices[clean_category_name] = valid_symptoms
        all_categorized_symptoms.update(valid_symptoms)

# Add truly uncategorized symptoms (already cleaned) to "Others"
uncategorized = sorted(list(feature_names_set - all_categorized_symptoms))
if uncategorized:
    original_choices["Others"] = uncategorized

# --- Visualization Functions (Refined Light Theme) ---
# create_prediction_chart, create_symptom_network, create_severity_gauge functions remain the same
def create_prediction_chart(diseases: List[str], probabilities: List[float]) -> go.Figure:
    """Create themed probability bar chart for light mode"""
    colors = ['#2a9d8f', '#e9c46a', '#f4a261', '#e76f51', '#d62828'] # Teal/Yellow/Orange/Red palette
    fig = go.Figure(data=[
        go.Bar(
            x=probabilities * 100,
            y=diseases,
            orientation='h',
            marker_color=colors[:len(diseases)],
            text=[f'{p:.1f}%' for p in probabilities * 100],
            textposition='auto', # Let Plotly decide best position
            textfont=dict(color='white', size=12, family='Segoe UI, sans-serif')
        )
    ])
    fig.update_layout(
        title={'text': 'Top 5 Predicted Conditions', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': 'var(--text-color-strong)'}},
        xaxis_title={'text': 'Probability (%)', 'font': {'size': 14, 'color': 'var(--text-color)'}},
        yaxis_title=None,
        xaxis=dict(
            range=[0, 100], tickmode='linear', tick0=0, dtick=20, ticksuffix='%',
            tickfont={'color': 'var(--text-color-muted)'},
            gridcolor='var(--border-color-light)', gridwidth=1
        ),
        yaxis=dict(
            tickfont={'color': 'var(--text-color)', 'size': 12},
            autorange="reversed"
        ),
        height=350,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Segoe UI, Roboto, sans-serif', 'color': 'var(--text-color)'}
    )
    fig.update_traces(textfont_color='black', selector=dict(marker_color='#e9c46a'))
    fig.update_traces(textfont_color='black', selector=dict(marker_color='#f4a261'))
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=False)
    return fig


# --- Helper Function (from new structure) ---
def _create_placeholder_figure(message: str) -> go.Figure:
    """Helper function to create a figure with a message."""
    fig = go.Figure()
    fig.update_layout(
        xaxis={'visible': False}, yaxis={'visible': False},
        annotations=[{
            'text': message,
            'xref': "paper", 'yref': "paper", 'showarrow': False,
            # Using CSS variable for color, provide fallback
            'font': {'size': 14, 'color': 'var(--text-color-muted, #888)'}
        }],
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20), height=400
    )
    return fig


# create symmptom network interface from the symptom_analysis.joblib
def create_symptom_network(
    selected_symptoms: List[str],
    symptom_analysis: Dict[str, Any],
    threshold: float = 0.3
    ) -> Optional[go.Figure]:
    """
    Create themed interactive network visualization with path highlighting.
    - Direct Selected<->Selected links: Green, thick
    - Indirect Paths between Selected: Yellow, medium thickness
    - Selected<->Related links (not part of paths): Grey, dotted
    - Related<->Related links: Light Grey, thin
    """

    # --- Input Validation ---
    if not selected_symptoms:
        return _create_placeholder_figure("Select symptoms to view relationships.")

    if 'correlation_matrix' not in symptom_analysis or symptom_analysis['correlation_matrix'].empty:
        return _create_placeholder_figure("Symptom correlation data unavailable.")

    corr_matrix = symptom_analysis['correlation_matrix']

    #get Feature Names for Check 
    feature_names_for_check = symptom_analysis.get('feature_names_original', list(corr_matrix.columns))
    feature_names_set_for_check = set(feature_names_for_check)

    # --- Network Construction 
    nodes = set()
    edges_data = []
    valid_selected_symptoms = [s for s in selected_symptoms if s in corr_matrix.columns]

    if not valid_selected_symptoms:
         return _create_placeholder_figure("Selected symptoms not found in correlation data (check names/spacing).")

    nodes.update(valid_selected_symptoms)
    initial_selection_set = set(valid_selected_symptoms) # Use valid ones for pathfinding

    for symptom in valid_selected_symptoms:
        try:
            correlations = corr_matrix[symptom]
            related_indices = correlations[correlations.abs() > threshold].index
            for rel in related_indices:
                if rel != symptom and rel in feature_names_set_for_check:
                    nodes.add(rel)
                    correlation_value = correlations[rel]
                    edge_tuple = (symptom, rel, abs(correlation_value))
                    # Store edges uniquely (e.g., sort nodes in tuple)
                    sorted_edge = tuple(sorted((symptom, rel)))
                    # Avoid adding duplicate edges if correlation is symmetric
                    if not any(e[0] == sorted_edge[0] and e[1] == sorted_edge[1] for e in edges_data):
                         edges_data.append( (sorted_edge[0], sorted_edge[1], abs(correlation_value)) )
        except KeyError:
             print(f"Warning: Symptom '{symptom}' unexpectedly caused KeyError during correlation lookup.")
             continue
        except Exception as e:
            print(f"Error processing symptom '{symptom}': {e}")
            continue

    # --- Check if graph is viable ---
    if len(nodes) <= 1 or not edges_data:
        # ... (placeholder logic remains the same) ...
        msg = f"No significant correlations > {threshold:.2f} found for the selected symptoms." if nodes else "Select symptoms."
        if valid_selected_symptoms and (len(nodes) <= 1 or not edges_data):
             msg = f"No significant correlations > {threshold:.2f} found for the selected symptoms."
        return _create_placeholder_figure(msg)

    # --- Build NetworkX Graph & Calculate Layout ---
    G = nx.Graph()
    node_list = list(nodes)
    G.add_nodes_from(node_list)
    edge_tuples_for_nx = [(u, v, {'weight': w}) for u, v, w in edges_data]
    G.add_edges_from(edge_tuples_for_nx)

    try:
        pos = nx.spring_layout(G, k=0.7, iterations=50, seed=42, weight='weight')
    except Exception as e: # Fallback layouts
        print(f"Spring layout failed ({e}), falling back to Kamada-Kawai.")
        try:
            pos = nx.kamada_kawai_layout(G, weight='weight')
        except Exception as e2:
            print(f"Kamada-Kawai layout failed ({e2}), falling back to random.")
            pos = nx.random_layout(G, seed=42)

    # --- Pathfinding and Edge Categorization ---
    direct_ss_edges = set()
    indirect_path_edges = set()

    # 1. Find direct selected-selected edges
    for u, v in G.edges():
        if u in initial_selection_set and v in initial_selection_set:
            direct_ss_edges.add(tuple(sorted((u, v))))

    # 2. Find shortest paths for non-directly connected selected pairs
    if len(initial_selection_set) >= 2:
        for node1, node2 in itertools.combinations(initial_selection_set, 2):
            # Check if nodes are in graph and path exists before calculating
            if G.has_node(node1) and G.has_node(node2) and not G.has_edge(node1, node2):
                try:
                    # Find shortest path (unweighted for simplicity)
                    path_nodes = nx.shortest_path(G, source=node1, target=node2)
                    # Extract edges from the path
                    for i in range(len(path_nodes) - 1):
                        u, v = path_nodes[i], path_nodes[i+1]
                        indirect_path_edges.add(tuple(sorted((u, v))))
                except nx.NetworkXNoPath:
                    # No path exists between these two selected nodes
                    pass
                except nx.NodeNotFound:
                     # Should not happen if G.has_node check passes, but good practice
                     pass

    # 3. Prepare coordinate lists for each edge type
    ss_edge_x, ss_edge_y = [], [] # Direct SS (Green)
    ip_edge_x, ip_edge_y = [], [] # Indirect Path (Yellow)
    sr_edge_x, sr_edge_y = [], [] # Selected-Related (Grey Dotted)
    rr_edge_x, rr_edge_y = [], [] # Related-Related (Light Grey)

    for u, v in G.edges():
        edge_tuple_sorted = tuple(sorted((u, v)))

        # Check if nodes exist in pos dictionary
        if u not in pos or v not in pos:
            print(f"Warning: Node position not found for edge ({u}, {v}). Skipping edge.")
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        coords = [x0, x1, None]
        coords_y = [y0, y1, None]

        # Assign edge to a category based on hierarchy
        if edge_tuple_sorted in direct_ss_edges:
            ss_edge_x.extend(coords)
            ss_edge_y.extend(coords_y)
        elif edge_tuple_sorted in indirect_path_edges:
             # Avoid drawing direct SS edges also as yellow path edges
             if edge_tuple_sorted not in direct_ss_edges:
                 ip_edge_x.extend(coords)
                 ip_edge_y.extend(coords_y)
        elif (u in initial_selection_set) ^ (v in initial_selection_set): # XOR: exactly one is selected
            sr_edge_x.extend(coords)
            sr_edge_y.extend(coords_y)
        else: # Neither is selected
            rr_edge_x.extend(coords)
            rr_edge_y.extend(coords_y)


    # --- Create Separate Edge Traces ---
    trace_rr = go.Scatter(x=rr_edge_x, y=rr_edge_y, line=dict(width=1, color='var(--border-color-light, #e9ecef)'), hoverinfo='none', mode='lines', name='Related Links')
    trace_sr = go.Scatter(x=sr_edge_x, y=sr_edge_y, line=dict(width=1.5, color='var(--text-color-muted, #888)', dash='dot'), hoverinfo='none', mode='lines', name='Selected-Related Links')
    trace_ip = go.Scatter(x=ip_edge_x, y=ip_edge_y, line=dict(width=2.5, color='var(--secondary-color, #e9c46a)'), hoverinfo='none', mode='lines', name='Indirect Path') # Yellow/Secondary
    trace_ss = go.Scatter(x=ss_edge_x, y=ss_edge_y, line=dict(width=3.5, color='var(--success-color, #28a745)'), hoverinfo='none', mode='lines', name='Direct Selected Links') # Green/Success

    # ... Node trace creation
    node_x = []
    node_y = []
    node_text_display = []
    node_hover_text = []
    node_color = []
    node_size = []
    node_border_color = []
    node_border_width = []

    # initial_selection_set = set(selected_symptoms) # Defined earlier

    for node in node_list: # Iterate in fixed order
        if node not in pos: # Robustness check for nodes
            print(f"Warning: Node position not found for node '{node}'. Skipping node.")
            continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        display_name = node.strip().replace('_', ' ')
        node_text_display.append(display_name)
        node_hover_text.append(f"Symptom: {node}<br>Connections: {G.degree(node)}")

        if node in initial_selection_set: # Use the set for checking
            node_color.append('var(--accent-color, #2a9d8f)') # Selected Node Color (Teal)
            node_size.append(22)
            node_border_color.append('var(--accent-dark, #217a70)')
            node_border_width.append(2)
        else:
            # Check if node is part of *any* indirect path between selected nodes
            is_on_indirect_path = any(node in edge for edge in indirect_path_edges)
            if is_on_indirect_path:
                 node_color.append('var(--secondary-dark, #d4ae5a)') # Darker Yellow for path nodes
                 node_size.append(14) # Slightly larger than default related
                 node_border_color.append('var(--secondary-color, #e9c46a)')
                 node_border_width.append(1.5)
            else:
                 node_color.append('var(--text-color-subtle, #adb5bd)') # Default Related Node Color (Grey)
                 node_size.append(12)
                 node_border_color.append('var(--border-color, #dee2e6)')
                 node_border_width.append(1)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', hoverinfo='text',
        hovertext=node_hover_text, text=node_text_display,
        textposition='top center', textfont=dict(size=9, color='var(--text-color, black)'),
        marker=dict(showscale=False, color=node_color, size=node_size,
                    line=dict(color=node_border_color, width=node_border_width))
    )
    # --- Final Figure Assembly (Order matters for drawing layers) ---
    fig = go.Figure(
        data=[trace_rr, trace_sr, trace_ip, trace_ss, node_trace], # Draw edges light->dark, nodes last
        layout=go.Layout(
             title=dict(text='Symptom Relationship Network', font=dict(size=16, color='var(--text-color-strong, black)'), x=0.5, xanchor='center'),
             showlegend=False, # Can turn on to see edge type names
             hovermode='closest',
             margin=dict(b=10, l=10, r=10, t=40),
             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=450
        )
    )

    return fig


# def create_symptom_network(selected_symptoms: List[str]) -> Optional[go.Figure]:
#     """Create themed interactive network visualization (light mode)"""
#     if not selected_symptoms or 'correlation_matrix' not in symptom_analysis or symptom_analysis['correlation_matrix'].empty:
#         fig = go.Figure()
#         fig.update_layout(
#             xaxis={'visible': False}, yaxis={'visible': False},
#             annotations=[{
#                 'text': "Select symptoms to view relationships." if not selected_symptoms else "Symptom correlation data unavailable.",
#                 'xref': "paper", 'yref': "paper", 'showarrow': False,
#                 'font': {'size': 14, 'color': 'var(--text-color-muted)'}
#             }],
#             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
#              margin=dict(l=20, r=20, t=40, b=20), height=400
#         )
#         return fig

#     corr_matrix = symptom_analysis['correlation_matrix']
#     threshold = 0.01

#     nodes = set(selected_symptoms)
#     edges_data = []
#     # Ensure we only check correlations for symptoms actually present in the matrix
#     # Use the cleaned feature names set for checking node validity
#     valid_symptoms_in_matrix = [s for s in selected_symptoms if s in corr_matrix.columns]

#     for symptom in valid_symptoms_in_matrix:
#         try:
#             correlations = corr_matrix[symptom]
#             related = correlations[correlations.abs() > threshold].index.tolist()
#             for rel_raw in related: # Correlation matrix likely has raw names
#                 rel_cleaned = rel_raw.strip() # Clean the related name
#                 if rel_cleaned != symptom and rel_cleaned in feature_names_set: # Check against cleaned set
#                     nodes.add(rel_cleaned) # Add cleaned name to nodes
#                     correlation_value = correlations[rel_raw]
#                     # Use cleaned names for edge tuple
#                     edge_tuple = tuple(sorted((symptom, rel_cleaned))) + (abs(correlation_value),)
#                     edges_data.append(edge_tuple)
#         except KeyError:
#              print(f"Warning: Symptom '{symptom}' not found in correlation matrix columns.")
#              continue

#     edges_data = [edge for edge in edges_data if edge[2] > 1e-6]

#     if len(nodes) <= 1 or not edges_data:
#          fig = go.Figure()
#          fig.update_layout(
#              xaxis={'visible': False}, yaxis={'visible': False},
#              annotations=[{
#                  'text': f"No significant symptom correlations found (threshold > {threshold}). Check correlation data." if nodes else "Select symptoms.",
#                  'xref': "paper", 'yref': "paper", 'showarrow': False,
#                  'font': {'size': 14, 'color': 'var(--text-color-muted)'}
#              }],
#              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
#              margin=dict(l=20, r=20, t=40, b=20), height=400
#          )
#          return fig

#     G = nx.Graph()
#     G.add_nodes_from(nodes) # Nodes are cleaned names
#     unique_edges = {edge[:2]: edge[2] for edge in edges_data} # Edges use cleaned names
#     G.add_weighted_edges_from([(u, v, w) for (u, v), w in unique_edges.items()])
#     pos = nx.spring_layout(G, k=0.8, iterations=70, seed=42)

#     edge_x, edge_y = [], []
#     for edge in G.edges():
#         x0, y0 = pos[edge[0]]
#         x1, y1 = pos[edge[1]]
#         edge_x.extend([x0, x1, None])
#         edge_y.extend([y0, y1, None])

#     edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='var(--border-color)'), opacity=0.7, hoverinfo='none', mode='lines')

#     node_x, node_y, node_text, node_colors, node_sizes = [], [], [], [], []
#     for node in G.nodes(): # Nodes are cleaned names
#         x, y = pos[node]
#         node_x.append(x)
#         node_y.append(y)
#         node_text.append(node.replace('_', ' ').title()) # Display cleaned name nicely
#         if node in selected_symptoms: # Check against selected (cleaned) symptoms
#             node_colors.append('var(--accent-color)')
#             node_sizes.append(28)
#         else:
#             node_colors.append('var(--secondary-color)')
#             node_sizes.append(18)

#     node_trace = go.Scatter(
#         x=node_x, y=node_y, mode='markers+text', hoverinfo='text', text=node_text,
#         textposition='bottom center', textfont=dict(size=11, color='var(--text-color)'),
#         marker=dict(showscale=False, color=node_colors, size=node_sizes, line_width=1.5, line_color='var(--background-card)')
#     )

#     fig = go.Figure(data=[edge_trace, node_trace],
#                  layout=go.Layout(
#                     title={'text': 'Symptom Relationship Network', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': 'var(--text-color-strong)'}},
#                     showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40),
#                     annotations=[dict(text=f"Lines indicate symptom correlation strength (threshold > {threshold}).", showarrow=False, align='left', xref="paper", yref="paper", x=0.01, y=-0.02, font=dict(size=10, color='var(--text-color-muted)'))],
#                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
#                     font={'family': 'Segoe UI, Roboto, sans-serif'}
#                     ))
#     return fig

def create_severity_gauge(severity_score: float) -> go.Figure:
    """Create themed severity gauge for light mode"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=severity_score,
        title={'text': "Calculated Severity Score", 'font': {'size': 18, 'color': 'var(--text-color-strong)'}},
        number={'suffix': "%", 'font': {'size': 28, 'color': 'var(--text-color-strong)'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "var(--text-color-muted)"},
            'bar': {'color': "var(--accent-color)", 'thickness': 0.4},
            'bgcolor': "rgba(0,0,0,0.05)", 'borderwidth': 1, 'bordercolor': "var(--border-color-light)",
            'steps': [
                {'range': [0, 30], 'color': '#a8d8b9'}, {'range': [30, 70], 'color': '#fde4a0'}, {'range': [70, 100], 'color': '#fab1a0'}
            ],
            'threshold': {'line': {'color': "#e17055", 'width': 4}, 'thickness': 0.75, 'value': severity_score}
        }))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Segoe UI, Roboto, sans-serif', 'color': 'var(--text-color)'},
        height=300, margin=dict(l=30, r=30, t=60, b=20)
    )
    return fig

# --- Prediction Logic ---
def calculate_severity_score(symptoms: List[str]) -> float:
    """Calculate severity score using the symptom severity dataset"""
    total_weight = 0
    count = 0
    try:
        max_possible_single_weight = max(severity_info.values()) if severity_info else 7
    except ValueError:
        max_possible_single_weight = 7

    for symptom in symptoms: # Expects cleaned symptom names
        lookup_symptom = symptom.lower().replace(' ', '_') # Convert to underscore format for severity dict
        weight = severity_info.get(lookup_symptom, 1)
        total_weight += weight
        count += 1

    if count == 0: return 0
    max_possible_total_weight = max_possible_single_weight * count
    severity_score = (total_weight / max_possible_total_weight) * 100 if max_possible_total_weight > 0 else 0
    return min(100, max(0, severity_score))

def predict_disease(*category_symptoms) -> Tuple[Optional[go.Figure], Optional[go.Figure], Optional[go.Figure], str, str]:
    """Predict disease, generate plots, and format output for refined light theme."""
    try:
        # Combine symptoms from all CheckboxGroup inputs, ensuring they are stripped
        all_symptoms_cleaned = list(set(s.strip() for sym_list in category_symptoms if isinstance(sym_list, list) for s in sym_list))

        # --- Debugging ---
        print("-" * 30)
        print(f"Symptoms received from UI (cleaned): {all_symptoms_cleaned}")
        # --- End Debugging ---

        if not all_symptoms_cleaned:
             empty_fig = go.Figure().update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis={'visible': False}, yaxis={'visible': False})
             placeholder_text = "<p style='color: var(--text-color-muted); text-align: center; padding: 20px;'>Please select symptoms to begin analysis.</p>"
             return empty_fig, create_symptom_network([]), create_severity_gauge(0), placeholder_text, ""

        # Create feature vector using the *original raw* feature names list
        feature_vector = pd.DataFrame(np.zeros((1, len(raw_feature_names))), columns=raw_feature_names)
        valid_symptoms_for_model = []
        unmatched_symptoms = []

        for cleaned_symptom in all_symptoms_cleaned:
            # Check if the cleaned symptom name exists in our cleaned set
            if cleaned_symptom in feature_names_set:
                 # Find the corresponding *original raw* feature name using the map
                 raw_name = cleaned_to_raw_map.get(cleaned_symptom)
                 if raw_name:
                     feature_vector[raw_name] = 1
                     valid_symptoms_for_model.append(cleaned_symptom) # Keep track of valid *cleaned* names
                 else:
                     # This shouldn't happen if map is built correctly
                     print(f"Error: Cleaned symptom '{cleaned_symptom}' in set but not in map.")
                     unmatched_symptoms.append(cleaned_symptom + " (mapping error)")
            else:
                unmatched_symptoms.append(cleaned_symptom)

        # --- Debugging ---
        print(f"Valid symptoms matched with model features (cleaned names): {valid_symptoms_for_model}")
        if unmatched_symptoms:
            print(f"Unmatched symptoms (not in feature_names_set or mapping error): {unmatched_symptoms}")
        # Example: Print non-zero entries in the feature vector
        # print(f"Feature vector non-zero columns: {feature_vector.columns[feature_vector.iloc[0] == 1].tolist()}")
        print("-" * 30)
        # --- End Debugging ---

        if not valid_symptoms_for_model:
             empty_fig = go.Figure().update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis={'visible': False}, yaxis={'visible': False})
             placeholder_text = "<p style='color: var(--text-color-muted); text-align: center; padding: 20px;'>Selected symptoms not recognized by the model.</p>"
             placeholder_text += f"<br><small style='color: var(--text-color-muted);'>Unmatched: {', '.join(unmatched_symptoms)}</small>" if unmatched_symptoms else ""
             return empty_fig, create_symptom_network([]), create_severity_gauge(0), placeholder_text, ""

        # --- Model Prediction ---
        probabilities = voting_clf.predict_proba(feature_vector)[0]
        top_5_idx = np.argsort(probabilities)[-5:][::-1]
        top_5_diseases = [target_names[i] for i in top_5_idx]
        top_5_probs = probabilities[top_5_idx]

        # --- Create Visualizations ---
        pred_chart = create_prediction_chart(top_5_diseases, top_5_probs)
        # Pass valid *cleaned* symptoms to network and severity functions
        symptom_net = create_symptom_network(valid_symptoms_for_model,symptom_analysis)
        severity = calculate_severity_score(valid_symptoms_for_model)
        severity_gauge = create_severity_gauge(severity)

        # --- Format Output ---
        top_disease = top_5_diseases[0]
        top_prob = top_5_probs[0]
        severity_level = '🔴 High' if severity > 70 else '🟡 Medium' if severity > 30 else '🟢 Low'

        main_output = f"""
        <div class="output-card">
            <h2 class="output-title">{top_disease.replace('_', ' ').title()}</h2>
            <p class="output-probability">Predicted Probability: <strong>{top_prob*100:.1f}%</strong></p>
            <p class="output-severity">Calculated Severity Level: <strong>{severity_level}</strong> ({severity:.1f}%)</p>
        </div>
        """

        detailed_output = "<h3 class='output-section-title'>Prediction Details (Top 5)</h3>"
        for i, (disease, prob) in enumerate(zip(top_5_diseases, top_5_probs)):
            try:
                d_desc = description_df.loc[description_df['Disease'] == disease, 'Description'].iloc[0]
            except IndexError:
                d_desc = "No description available."
            try:
                d_prec_row = precaution_df[precaution_df['Disease'] == disease].iloc[0]
                d_precautions = [p for p in d_prec_row[1:] if pd.notna(p)]
            except IndexError:
                d_precautions = ["No precautions available."]

            detailed_output += f"""
            <details class="output-accordion">
                <summary>
                    <span class="accordion-rank">{i+1}.</span> {disease.replace('_', ' ').title()}
                    <span class="accordion-prob">({prob*100:.1f}%)</span>
                </summary>
                <div class="accordion-content">
                    <h5 class="content-subtitle">Description:</h5>
                    <p>{d_desc}</p>
                    <h5 class="content-subtitle">Precautions:</h5>
                    <ul>{''.join([f"<li>{prec.strip().capitalize()}</li>" for prec in d_precautions])}</ul>
                </div>
            </details>
            """

        return pred_chart, symptom_net, severity_gauge, main_output, detailed_output

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        traceback.print_exc()
        empty_fig = go.Figure().update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis={'visible': False}, yaxis={'visible': False})
        error_message = f"<p style='color: #e74c3c; text-align: center; padding: 20px;'>An error occurred during analysis. Please check logs.</p>"
        return empty_fig, empty_fig, empty_fig, error_message, ""

# --- Search Logic ---
def filter_symptoms(search_term: str, *current_values: List[List[str]]):
    """Filters symptom choices within each category tab based on search term."""
    updates = []
    search_lower = search_term.lower().strip() if search_term else ""
    category_keys = list(original_choices.keys()) # Get ordered keys

    if len(current_values) != len(category_keys):
        print(f"Warning: Mismatch between current_values ({len(current_values)}) and categories ({len(category_keys)}). Search might not preserve selections correctly.")
        safe_current_values = (list(current_values) + [[]] * len(category_keys))[:len(category_keys)]
    else:
        safe_current_values = [cv if isinstance(cv, list) else [] for cv in current_values]

    for idx, category_name in enumerate(category_keys):
        if category_name in original_choices:
            original_category_choices = original_choices[category_name]
            filtered_choices = [s for s in original_category_choices if search_lower in s.lower().replace('_', ' ')]
            current_selection = safe_current_values[idx]
            preserved_value = [v for v in current_selection if v in filtered_choices]
            updates.append(gr.update(choices=filtered_choices, value=preserved_value))
        else:
             print(f"Warning: Category '{category_name}' not found in original_choices during search update for index {idx}.")
             updates.append(gr.update())

    return updates


# --- Custom CSS for Refined Light Theme ---
custom_css = """
:root {
    --accent-color: #2a9d8f; /* Teal accent */
    --accent-dark: #217a70;
    --secondary-color: #e9c46a; /* Soft Yellow */
    --secondary-dark: #d4ae5a;
    --background-app: #f8f9fa; /* Very light grey background */
    --background-card: #ffffff; /* White cards */
    --background-input: #ffffff;
    --background-input-hover: #f1f3f5;
    --text-color: #495057; /* Dark grey text */
    --text-color-strong: #212529; /* Black text */
    --text-color-muted: #6c757d; /* Medium grey text */
    --text-color-subtle: #adb5bd; /* Light grey text */
    --border-color: #dee2e6; /* Light border */
    --border-color-light: #e9ecef;
    --card-shadow: 0 3px 10px rgba(0,0,0,0.05);
    --input-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

body {
    font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background-color: var(--background-app);
    color: var(--text-color);
    line-height: 1.6;
}

.gradio-container {
    border-radius: 10px !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.07) !important;
    background-color: var(--background-card);
    border: 1px solid var(--border-color);
    overflow: hidden;
}

/* Titles and Headers */
h1, h2, h3, h4, h5 { color: var(--text-color-strong); font-weight: 600; }
#main-title {
    text-align: center;
    color: var(--accent-color);
    margin-bottom: 5px !important;
    font-size: 2.1em;
    font-weight: 700;
    padding-top: 20px;
}
#sub-title {
    text-align: center;
    color: var(--text-color-muted);
    margin-bottom: 30px !important;
    font-size: 1.1em;
}

/* Buttons */
.gr-button {
    background: linear-gradient(180deg, var(--accent-color) 0%, var(--accent-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 18px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    font-weight: 500;
}
.gr-button:hover {
    background: linear-gradient(180deg, var(--accent-dark) 0%, var(--accent-color) 100%) !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    transform: translateY(-1px);
}
.gr-button.secondary {
    background: linear-gradient(180deg, #ced4da 0%, #adb5bd 100%) !important; /* Light grey gradient */
    color: var(--text-color-strong) !important;
}
.gr-button.secondary:hover {
    background: linear-gradient(180deg, #adb5bd 0%, #ced4da 100%) !important;
}

/* Input Areas */
#symptom-column {
    background-color: var(--background-app); /* Match app background */
    padding: 20px;
    border-radius: 8px;
    border: 1px solid var(--border-color-light);
    margin-right: 10px;
}
#symptom-search-box textarea {
    border-radius: 6px !important;
    border: 1px solid var(--border-color) !important;
    background-color: var(--background-input) !important;
    color: var(--text-color) !important;
    padding: 9px 12px !important;
    margin-bottom: 15px !important;
    box-shadow: var(--input-shadow);
}
#symptom-search-box textarea:focus {
     border-color: var(--accent-color) !important;
     box-shadow: 0 0 0 2px rgba(42, 157, 143, 0.2); /* Focus ring */
}
#symptom-search-box textarea::placeholder {
    color: var(--text-color-muted);
}
.gr-tabs > .tab-nav > button {
    padding: 10px 15px !important;
    border-radius: 6px 6px 0 0 !important;
    background-color: var(--background-app) !important; /* Match app background */
    color: var(--text-color-muted) !important;
    border-bottom: 2px solid transparent !important;
    transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    font-weight: 500;
    margin-right: 4px;
}
.gr-tabs > .tab-nav > button.selected {
    background-color: var(--background-card) !important; /* White selected tab */
    color: var(--accent-color) !important;
    border-bottom: 2px solid var(--accent-color) !important;
}
.gr-checkbox-group {
    background-color: var(--background-card);
    border: 1px solid var(--border-color-light);
    border-radius: 6px;
    padding: 12px;
    max-height: 350px;
    overflow-y: auto;
}
.gr-checkbox-label span {
    color: var(--text-color);
}
.gr-checkbox-label input[type=checkbox]:checked + span {
    color: var(--accent-dark);
    font-weight: 500;
}

/* Output Areas */
#results-column {
    padding: 20px;
    margin-left: 10px;
}
#results-tabs > .tab-nav > button { /* Style result tabs like input tabs */
     padding: 10px 15px !important;
    border-radius: 6px 6px 0 0 !important;
    background-color: var(--background-app) !important;
    color: var(--text-color-muted) !important;
    border-bottom: 2px solid transparent !important;
    transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    font-weight: 500;
     margin-right: 4px;
}
#results-tabs > .tab-nav > button.selected {
    background-color: var(--background-card) !important;
    color: var(--accent-color) !important;
    border-bottom: 2px solid var(--accent-color) !important;
}
.gr-plot {
    border: 1px solid var(--border-color-light);
    border-radius: 8px;
    padding: 10px;
    background-color: var(--background-card);
    box-shadow: var(--card-shadow);
    margin-top: 15px;
}
/* Plotly styling adjustments for light theme */
.js-plotly-plot .plotly .gridlayer path { stroke: var(--border-color-light) !important; opacity: 0.7;}
.js-plotly-plot .plotly .tick text { fill: var(--text-color-muted) !important; }
.js-plotly-plot .plotly .axistext text { fill: var(--text-color) !important; }
.js-plotly-plot .plotly .annotation text { fill: var(--text-color-muted) !important; }
.js-plotly-plot .plotly .indicator text { fill: var(--text-color-strong) !important; }
.js-plotly-plot .plotly .indicator .gauge-axis line,
.js-plotly-plot .plotly .indicator .gauge-axis path { stroke: var(--text-color-muted) !important; }
.js-plotly-plot .plotly .indicator .gauge-bar { fill: var(--accent-color) !important; }

/* Output Card & Accordion Styling */
.output-card {
    padding: 15px 20px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    background-color: var(--background-card);
    box-shadow: var(--card-shadow);
    margin-bottom: 15px; /* Space below card */
}
.output-title { margin-top: 0; color: var(--accent-dark); font-size: 1.4em; }
.output-probability { font-size: 1.1em; margin-bottom: 10px; color: var(--text-color-strong); }
.output-severity { color: var(--text-color-muted); }

.output-section-title { margin-top: 0; color: var(--accent-dark); border-bottom: 1px solid var(--border-color-light); padding-bottom: 5px; margin-bottom: 15px; }

.output-accordion {
    margin-bottom: 12px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    overflow: hidden;
    background-color: var(--background-card);
}
.output-accordion summary {
    cursor: pointer;
    font-weight: 600;
    color: var(--text-color-strong);
    background-color: var(--background-app); /* Slightly different background for summary */
    padding: 12px 15px;
    border-bottom: 1px solid var(--border-color);
    transition: background-color 0.2s ease;
    position: relative;
}
.output-accordion summary:hover {
    background-color: var(--background-input-hover);
}
.output-accordion[open] summary {
    border-bottom: 1px solid var(--border-color);
}
.output-accordion summary::marker { /* Style default arrow */
    color: var(--accent-color);
}
.accordion-rank {
    display: inline-block;
    width: 25px; /* Fixed width for rank number */
    color: var(--accent-color);
}
.accordion-prob {
    float: right;
    font-weight: normal;
    color: var(--text-color-muted);
    font-size: 0.95em;
}
.accordion-content {
    padding: 15px;
    background-color: var(--background-card);
}
.content-subtitle {
    margin-bottom: 5px;
    color: var(--accent-dark);
    font-weight: 500;
    font-size: 1em;
}
.accordion-content p, .accordion-content ul {
    margin-top: 5px;
    font-size: 0.95em;
    color: var(--text-color);
}
.accordion-content ul { padding-left: 20px; }

#severity-gauge-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px 0;
    background-color: var(--background-card);
    border-radius: 8px;
    border: 1px solid var(--border-color-light);
    box-shadow: var(--card-shadow);
    margin-top: 15px;
}
"""

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="yellow"), css=custom_css) as iface:
    gr.Markdown("# 🩺 AI Health Predictor & Advisor", elem_id="main-title")
    gr.Markdown("Select symptoms for analysis and insights.", elem_id="sub-title")

    with gr.Row(): # Main layout row
        # --- Input Column ---
        with gr.Column(scale=1, elem_id="symptom-column"):
            gr.Markdown("### 1. Select Your Symptoms")
            symptom_search = gr.Textbox(
                label="Search Symptoms",
                placeholder="Type to filter symptoms...",
                elem_id="symptom-search-box",
                show_label=False
            )
            # --- Corrected Tab and CheckboxGroup Creation ---
            category_inputs = [] # This list will hold the CheckboxGroup components
            with gr.Tabs():
                category_keys = list(original_choices.keys()) # Get ordered keys
                for category in category_keys:
                    # Use the cleaned symptoms from original_choices for display
                    valid_symptoms_for_display = original_choices[category]
                    with gr.Tab(f"{category}"):
                        category_input = gr.CheckboxGroup(
                            choices=valid_symptoms_for_display, # Show cleaned names
                            label=None,
                            value=[],
                        )
                        category_inputs.append(category_input)
            # --- End Corrected Tab Creation ---

            with gr.Row():
                 predict_btn = gr.Button("Analyze Symptoms", variant="primary", elem_id="analyze-button")
                 clear_btn = gr.Button("Clear All", variant="secondary", elem_id="clear-button")

        # --- Output Column ---
        with gr.Column(scale=2, elem_id="results-column"):
            gr.Markdown("### 2. Analysis Results")
            with gr.Tabs(elem_id="results-tabs") as results_tabs:
                with gr.Tab("Top Prediction", id=0):
                    output_main = gr.Markdown(elem_id="output-box") # Summary card
                    prob_plot = gr.Plot(label="Top 5 Probabilities") # Bar chart

                with gr.Tab("Detailed Predictions", id=1):
                     output_detailed = gr.Markdown(elem_id="detailed-predictions-box") # Accordion

                with gr.Tab("Severity Score", id=2):
                    severity_plot = gr.Plot(label="Severity Score", elem_id="severity-gauge-container")

                with gr.Tab("Symptom Network", id=3):
                    network_plot = gr.Plot(label="Related Symptoms Network")


    # --- Event Handlers ---
    predict_btn.click(
        fn=predict_disease,
        inputs=category_inputs, # Pass the list of CheckboxGroup components
        outputs=[
            prob_plot,
            network_plot,
            severity_plot,
            output_main,
            output_detailed
        ]
    )

    # Clear button logic
    def clear_all_inputs_and_search():
        # Reset search box and all checkbox groups
        # Also reset the symptom choices to their original full lists
        updates = [gr.update(value="")] # Clear search box
        # Use category_keys which has the correct order matching category_inputs
        for category_name in category_keys:
             updates.append(gr.update(choices=original_choices[category_name], value=[]))
        return updates

    clear_btn.click(
        fn=clear_all_inputs_and_search,
        inputs=None,
        outputs=[symptom_search] + category_inputs # Clear search and checkboxes
    )

    # Search event handler - Ensure inputs/outputs match the filter_symptoms function
    symptom_search.change(
        fn=filter_symptoms,
        inputs=[symptom_search] + category_inputs, # Pass search term and current values of *all* groups
        outputs=category_inputs # Update *all* checkbox groups
    )


# --- Launch ---
if __name__ == "__main__":
    iface.launch(share=True) # Set share=False for local use
