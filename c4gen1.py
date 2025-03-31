import streamlit as st
import re
import base64
import tempfile
import os
import json

def clean_id(text):
    """Clean text to create valid Mermaid IDs"""
    return re.sub(r'[^a-zA-Z0-9]', '', text)


def generate_html_with_mermaid(mermaid_code):
    """Generate HTML that renders the Mermaid diagram"""
    html_content = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>C4 Model Diagram</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .mermaid {{
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>C4 Model Diagram</h1>
        <div class="mermaid">
{mermaid_code}
        </div>
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true }});
    </script>
</body>
</html>"""
    return html_content

def parse_c4_file(uploaded_file):
    try:
        content = uploaded_file.read().decode("utf-8")
        data = json.loads(content)

        if 'systems' in data:
            for system in data['systems']:
                system['id'] = clean_id(system['name'])
            st.session_state.systems = data['systems']

        if 'persons' in data:
            for person in data['persons']:
                person['id'] = clean_id(person['name'])
            st.session_state.persons = data['persons']

        if 'containers' in data:
            st.session_state.containers = data['containers']

        if 'components' in data:
            st.session_state.components = data['components']

        if 'relationships' in data:
            st.session_state.relationships = data['relationships']

        st.success("C4 model loaded successfully from file!")
    except Exception as e:
        st.error(f"Failed to load C4 file: {e}")

def create_html_file(mermaid_code):
    """Create a temporary HTML file with the Mermaid diagram"""
    html_content = generate_html_with_mermaid(mermaid_code)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
        f.write(html_content)
        return f.name


def get_html_download_link(mermaid_code, filename="c4_diagram.html"):
    """Generate a download link for the HTML file"""
    html_content = generate_html_with_mermaid(mermaid_code)
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f'data:text/html;base64,{b64}'
    return f'<a href="{href}" download="{filename}">Download HTML File</a>'


def main():
    st.set_page_config(page_title="C4 Model Diagram Generator", layout="wide")

    st.title("C4 Model Diagram Generator")
    st.markdown("""
    This application helps you create C4 model diagrams using Mermaid.js.
    The C4 model provides a way to visualize software architecture at different levels of abstraction.
    """)

    # Upload C4 JSON file
    uploaded_file = st.file_uploader("Upload C4 JSON File", type=["json"])
    if uploaded_file:
        parse_c4_file(uploaded_file)

    # Initialize session state variables if they don't exist
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'systems' not in st.session_state:
        st.session_state.systems = []
    if 'containers' not in st.session_state:
        st.session_state.containers = {}
    if 'components' not in st.session_state:
        st.session_state.components = {}
    if 'persons' not in st.session_state:
        st.session_state.persons = []
    if 'relationships' not in st.session_state:
        st.session_state.relationships = []
    if 'current_mermaid_code' not in st.session_state:
        st.session_state.current_mermaid_code = ""
    if 'show_popup' not in st.session_state:
        st.session_state.show_popup = False

    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        if st.button("1. Context Diagram"):
            st.session_state.step = 1
        if st.button("2. Container Diagram"):
            st.session_state.step = 2
        if st.button("3. Component Diagram"):
            st.session_state.step = 3
        if st.button("4. Relationships"):
            st.session_state.step = 4
        if st.button("5. Generate Diagram"):
            st.session_state.step = 5

    # Main area based on current step
    if st.session_state.step == 1:
        context_diagram()
    elif st.session_state.step == 2:
        container_diagram()
    elif st.session_state.step == 3:
        component_diagram()
    elif st.session_state.step == 4:
        relationships()
    elif st.session_state.step == 5:
        generate_diagram()

    # Show popup if enabled
    if st.session_state.show_popup:
        show_diagram_popup()


def context_diagram():
    st.header("Step 1: Context Diagram")
    st.markdown("""
    In this step, define the systems and persons (users) in your enterprise architecture.
    The context diagram shows the big picture of your system and how it interacts with users and external systems.
    """)

    # Systems
    st.subheader("Systems")
    st.markdown("Define the software systems in your enterprise architecture.")

    # Display existing systems
    if st.session_state.systems:
        st.write("Current Systems:")
        for i, system in enumerate(st.session_state.systems):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{i + 1}. {system['name']} - {system['description']}")
            with col2:
                if st.button(f"Remove System {i + 1}"):
                    st.session_state.systems.pop(i)
                    st.rerun()

    # Add new system
    with st.form("add_system_form"):
        st.write("Add a new system:")
        system_name = st.text_input("System Name")
        system_description = st.text_area("System Description")
        system_type = st.selectbox("System Type", ["Internal", "External"])
        submit_system = st.form_submit_button("Add System")

        if submit_system and system_name and system_description:
            st.session_state.systems.append({
                "name": system_name,
                "description": system_description,
                "type": system_type,
                "id": clean_id(system_name)
            })
            st.success(f"System '{system_name}' added successfully!")
            st.rerun()

    # Persons
    st.subheader("Persons (Users)")
    st.markdown("Define the users or actors that interact with your systems.")

    # Display existing persons
    if st.session_state.persons:
        st.write("Current Persons:")
        for i, person in enumerate(st.session_state.persons):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{i + 1}. {person['name']} - {person['description']}")
            with col2:
                if st.button(f"Remove Person {i + 1}"):
                    st.session_state.persons.pop(i)
                    st.rerun()

    # Add new person
    with st.form("add_person_form"):
        st.write("Add a new person:")
        person_name = st.text_input("Person Name")
        person_description = st.text_area("Person Description")
        submit_person = st.form_submit_button("Add Person")

        if submit_person and person_name and person_description:
            st.session_state.persons.append({
                "name": person_name,
                "description": person_description,
                "id": clean_id(person_name)
            })
            st.success(f"Person '{person_name}' added successfully!")
            st.rerun()

    # Navigation
    col1, col2 = st.columns(2)
    with col2:
        if st.button("Next: Container Diagram", key="next_to_containers"):
            st.session_state.step = 2
            st.rerun()


def container_diagram():
    st.header("Step 2: Container Diagram")
    st.markdown("""
    Define containers for each system. Containers are runtime units (applications, data stores, etc.) 
    that make up a system.
    """)

    if not st.session_state.systems:
        st.warning("Please add at least one system in the Context Diagram step before proceeding.")
        if st.button("Go back to Context Diagram"):
            st.session_state.step = 1
            st.rerun()
        return

    # Select system
    system_names = [system["name"] for system in st.session_state.systems if system["type"] == "Internal"]
    if not system_names:
        st.warning("You need at least one internal system to define containers.")
        if st.button("Go back to Context Diagram"):
            st.session_state.step = 1
            st.rerun()
        return

    selected_system = st.selectbox("Select System", system_names)
    system_id = clean_id(selected_system)

    if system_id not in st.session_state.containers:
        st.session_state.containers[system_id] = []

    # Display existing containers
    if st.session_state.containers[system_id]:
        st.write(f"Current Containers for {selected_system}:")
        for i, container in enumerate(st.session_state.containers[system_id]):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(
                    f"{i + 1}. {container['name']} - {container['description']} (Technology: {container['technology']})")
            with col2:
                if st.button(f"Remove Container {i + 1}"):
                    st.session_state.containers[system_id].pop(i)
                    st.rerun()

    # Add new container
    with st.form(f"add_container_form_{system_id}"):
        st.write(f"Add a new container to {selected_system}:")
        container_name = st.text_input("Container Name")
        container_description = st.text_area("Container Description")
        container_technology = st.text_input("Technology (e.g., Spring Boot, PostgreSQL)")
        submit_container = st.form_submit_button("Add Container")

        if submit_container and container_name and container_description:
            st.session_state.containers[system_id].append({
                "name": container_name,
                "description": container_description,
                "technology": container_technology,
                "id": f"{system_id}_{clean_id(container_name)}"
            })
            st.success(f"Container '{container_name}' added to '{selected_system}' successfully!")
            st.rerun()

    # Navigation
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Previous: Context Diagram", key="prev_to_context"):
            st.session_state.step = 1
            st.rerun()
    with col3:
        if st.button("Next: Component Diagram", key="next_to_components"):
            st.session_state.step = 3
            st.rerun()


def component_diagram():
    st.header("Step 3: Component Diagram")
    st.markdown("""
    Define components for each container. Components are grouped chunks of code 
    (modules, packages, etc.) within a container.
    """)

    # Check if containers exist
    has_containers = False
    for system_id, containers in st.session_state.containers.items():
        if containers:
            has_containers = True
            break

    if not has_containers:
        st.warning("Please add at least one container in the Container Diagram step before proceeding.")
        if st.button("Go back to Container Diagram"):
            st.session_state.step = 2
            st.rerun()
        return

    # Select system
    system_names = [system["name"] for system in st.session_state.systems if system["type"] == "Internal"]
    selected_system = st.selectbox("Select System", system_names)
    system_id = clean_id(selected_system)

    if system_id not in st.session_state.containers or not st.session_state.containers[system_id]:
        st.warning(f"No containers defined for {selected_system}. Please add containers first.")
        if st.button("Go back to Container Diagram"):
            st.session_state.step = 2
            st.rerun()
        return

    # Select container
    container_names = [container["name"] for container in st.session_state.containers[system_id]]
    selected_container = st.selectbox("Select Container", container_names)
    container_id = f"{system_id}_{clean_id(selected_container)}"

    if container_id not in st.session_state.components:
        st.session_state.components[container_id] = []

    # Display existing components
    if st.session_state.components[container_id]:
        st.write(f"Current Components for {selected_container}:")
        for i, component in enumerate(st.session_state.components[container_id]):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(
                    f"{i + 1}. {component['name']} - {component['description']} (Technology: {component['technology']})")
            with col2:
                if st.button(f"Remove Component {i + 1}"):
                    st.session_state.components[container_id].pop(i)
                    st.rerun()

    # Add new component
    with st.form(f"add_component_form_{container_id}"):
        st.write(f"Add a new component to {selected_container}:")
        component_name = st.text_input("Component Name")
        component_description = st.text_area("Component Description")
        component_technology = st.text_input("Technology (e.g., Spring MVC, React)")
        submit_component = st.form_submit_button("Add Component")

        if submit_component and component_name and component_description:
            st.session_state.components[container_id].append({
                "name": component_name,
                "description": component_description,
                "technology": component_technology,
                "id": f"{container_id}_{clean_id(component_name)}"
            })
            st.success(f"Component '{component_name}' added to '{selected_container}' successfully!")
            st.rerun()

    # Navigation
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Previous: Container Diagram", key="prev_to_containers"):
            st.session_state.step = 2
            st.rerun()
    with col3:
        if st.button("Next: Relationships", key="next_to_relationships"):
            st.session_state.step = 4
            st.rerun()


def relationships():
    st.header("Step 4: Relationships")
    st.markdown("""
    Define relationships between elements in your C4 model. 
    Relationships show how different parts of your architecture interact with each other.
    """)

    # Display existing relationships
    if st.session_state.relationships:
        st.subheader("Current Relationships:")
        for i, rel in enumerate(st.session_state.relationships):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{i + 1}. {rel['source_name']} → {rel['target_name']}: {rel['description']}")
            with col2:
                if st.button(f"Remove Relationship {i + 1}"):
                    st.session_state.relationships.pop(i)
                    st.rerun()

    # Add new relationship
    st.subheader("Add a new relationship:")
    with st.form("add_relationship_form"):
        # Prepare options for sources and targets
        source_options = []
        # Add persons
        for person in st.session_state.persons:
            source_options.append({
                "name": f"Person: {person['name']}",
                "id": person['id'],
                "type": "person"
            })
        # Add systems
        for system in st.session_state.systems:
            source_options.append({
                "name": f"System: {system['name']}",
                "id": system['id'],
                "type": "system"
            })
        # Add containers
        for system_id, containers in st.session_state.containers.items():
            for container in containers:
                source_options.append({
                    "name": f"Container: {container['name']}",
                    "id": container['id'],
                    "type": "container"
                })
        # Add components
        for container_id, components in st.session_state.components.items():
            for component in components:
                source_options.append({
                    "name": f"Component: {component['name']}",
                    "id": component['id'],
                    "type": "component"
                })

        source_names = [option["name"] for option in source_options]
        target_names = source_names.copy()

        source_selected = st.selectbox("Source", source_names)
        target_selected = st.selectbox("Target", target_names)
        relationship_description = st.text_input("Relationship Description (e.g., 'uses', 'sends data to')")
        submit_relationship = st.form_submit_button("Add Relationship")

        if submit_relationship and source_selected and target_selected and relationship_description:
            source_index = source_names.index(source_selected)
            target_index = target_names.index(target_selected)

            source = source_options[source_index]
            target = source_options[target_index]

            if source["id"] != target["id"]:
                st.session_state.relationships.append({
                    "source_id": source["id"],
                    "source_name": source_selected,
                    "target_id": target["id"],
                    "target_name": target_selected,
                    "description": relationship_description
                })
                st.success(f"Relationship from '{source_selected}' to '{target_selected}' added successfully!")
                st.rerun()
            else:
                st.error("Source and target cannot be the same!")

    # Navigation
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Previous: Component Diagram", key="prev_to_components"):
            st.session_state.step = 3
            st.rerun()
    with col3:
        if st.button("Next: Generate Diagram", key="next_to_generate"):
            st.session_state.step = 5
            st.rerun()


def generate_diagram():
    st.header("Step 5: Generate Diagram")

    diagram_type = st.selectbox("Select Diagram Type", ["Context", "Container", "Component"])
    selected_system = None
    selected_container = None

    if diagram_type in ["Container", "Component"]:
        system_names = [system["name"] for system in st.session_state.systems if system["type"] == "Internal"]
        if system_names:
            selected_system = st.selectbox("Select System", system_names)
        else:
            st.warning("No internal systems available to generate Container or Component diagrams.")
            return

    if diagram_type == "Component" and selected_system:
        system_id = clean_id(selected_system)
        if system_id in st.session_state.containers and st.session_state.containers[system_id]:
            container_names = [container["name"] for container in st.session_state.containers[system_id]]
            selected_container = st.selectbox("Select Container", container_names)
        else:
            st.warning(f"No containers available for {selected_system} to generate Component diagram.")
            return

    # Generate Mermaid diagram
    mermaid_code = generate_mermaid_code(diagram_type, selected_system, selected_container)
    st.session_state.current_mermaid_code = mermaid_code

    # Display diagram
    st.subheader("Generated C4 Model Diagram")
    st.markdown(f"```mermaid\n{mermaid_code}\n```")

    # Show raw code
    with st.expander("Show Raw Mermaid Code"):
        st.code(mermaid_code, language="javascript")

    # Download options
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            label="Download Mermaid Code",
            data=mermaid_code,
            file_name="c4_model_diagram.mmd",
            mime="text/plain"
        )

    with col2:
        st.markdown(get_html_download_link(mermaid_code), unsafe_allow_html=True)

    with col3:
        if st.button("Open Interactive Viewer"):
            st.session_state.show_popup = True
            st.rerun()

    # Navigation
    if st.button("Previous: Relationships", key="prev_to_relationships"):
        st.session_state.step = 4
        st.rerun()


def show_diagram_popup():
    """Show a popup with an interactive Mermaid diagram"""
    # Create a centered container for the popup
    popup_container = st.container()

    with popup_container:
        st.markdown("<h2 style='text-align: center;'>Interactive C4 Model Diagram</h2>", unsafe_allow_html=True)

        # Create an iframe that loads the HTML with Mermaid
        # We use the HTML iframe approach since Streamlit doesn't support direct JavaScript execution
        html_file = create_html_file(st.session_state.current_mermaid_code)

        # Create a data URL for the iframe
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Create a base64 encoded data URL
        b64 = base64.b64encode(html_content.encode()).decode()
        iframe_src = f"data:text/html;base64,{b64}"

        # Display the iframe with the Mermaid diagram
        st.markdown(f"""
        <div style="display: flex; justify-content: center; width: 100%;">
            <iframe src="{iframe_src}" width="100%" height="600px" style="border: none;"></iframe>
        </div>
        """, unsafe_allow_html=True)

        # Close button
        if st.button("Close Viewer"):
            st.session_state.show_popup = False
            # Clean up the temporary file
            if os.path.exists(html_file):
                os.unlink(html_file)
            st.rerun()


def generate_mermaid_code(diagram_type, selected_system=None, selected_container=None):
    """Generate Mermaid code based on the data and selected diagram type"""
    mermaid_code = "C4Context\n"

    # Title based on diagram type
    if diagram_type == "Context":
        mermaid_code += "    title Context Diagram\n"
    elif diagram_type == "Container" and selected_system:
        mermaid_code += f"    title Container Diagram for {selected_system}\n"
    elif diagram_type == "Component" and selected_system and selected_container:
        mermaid_code += f"    title Component Diagram for {selected_container} in {selected_system}\n"

    # Add elements based on diagram type
    # Persons
    for person in st.session_state.persons:
        mermaid_code += f"    Person({person['id']}, \"{person['name']}\", \"{person['description']}\")\n"

    # Systems
    if diagram_type == "Context":
        for system in st.session_state.systems:
            if system["type"] == "Internal":
                mermaid_code += f"    System({system['id']}, \"{system['name']}\", \"{system['description']}\")\n"
            else:
                mermaid_code += f"    System_Ext({system['id']}, \"{system['name']}\", \"{system['description']}\")\n"

    # Containers
    system_id = None
    if selected_system:
        system_id = clean_id(selected_system)

    if diagram_type in ["Container", "Component"] and system_id:
        # Add the system boundary
        system_obj = next((s for s in st.session_state.systems if s["id"] == system_id), None)
        if system_obj:
            mermaid_code += f"    System_Boundary({system_id}, \"{system_obj['name']}\")"

            # Add containers for the selected system
            if system_id in st.session_state.containers:
                for container in st.session_state.containers[system_id]:
                    if container["technology"]:
                        mermaid_code += f"    Container({container['id']}, \"{container['name']}\", \"{container['technology']}\", \"{container['description']}\")\n"
                    else:
                        mermaid_code += f"    Container({container['id']}, \"{container['name']}\", \"{container['description']}\")\n"

    # Components
    container_id = None
    if selected_system and selected_container:
        container_id = f"{system_id}_{clean_id(selected_container)}"

    if diagram_type == "Component" and container_id:
        # Add the container boundary
        container_obj = next((c for c in st.session_state.containers[system_id] if c["id"] == container_id), None)
        if container_obj:
            mermaid_code += f"    Container_Boundary({container_id}, \"{container_obj['name']}\")"

            # Add components for the selected container
            if container_id in st.session_state.components:
                for component in st.session_state.components[container_id]:
                    if component["technology"]:
                        mermaid_code += f"    Component({component['id']}, \"{component['name']}\", \"{component['technology']}\", \"{component['description']}\")\n"
                    else:
                        mermaid_code += f"    Component({component['id']}, \"{component['name']}\", \"{component['description']}\")\n"

    # Add relationships based on the diagram type
    for rel in st.session_state.relationships:
        # For context diagram, show only relationships involving persons and systems
        if diagram_type == "Context":
            source_is_valid = any(p["id"] == rel["source_id"] for p in st.session_state.persons) or any(
                s["id"] == rel["source_id"] for s in st.session_state.systems)
            target_is_valid = any(p["id"] == rel["target_id"] for p in st.session_state.persons) or any(
                s["id"] == rel["target_id"] for s in st.session_state.systems)

            if source_is_valid and target_is_valid:
                mermaid_code += f"    Rel({rel['source_id']}, {rel['target_id']}, \"{rel['description']}\")\n"

        # For container diagram, include relationships involving the selected system's containers
        elif diagram_type == "Container" and system_id:
            # Check if either source or target is in the selected system
            source_in_system = rel["source_id"] == system_id or rel["source_id"].startswith(f"{system_id}_")
            target_in_system = rel["target_id"] == system_id or rel["target_id"].startswith(f"{system_id}_")

            if source_in_system or target_in_system:
                mermaid_code += f"    Rel({rel['source_id']}, {rel['target_id']}, \"{rel['description']}\")\n"

        # For component diagram, include relationships involving the selected container's components
        elif diagram_type == "Component" and container_id:
            # Check if either source or target is in the selected container
            source_in_container = rel["source_id"] == container_id or rel["source_id"].startswith(f"{container_id}_")
            target_in_container = rel["target_id"] == container_id or rel["target_id"].startswith(f"{container_id}_")

            if source_in_container or target_in_container:
                mermaid_code += f"    Rel({rel['source_id']}, {rel['target_id']}, \"{rel['description']}\")\n"

    return mermaid_code


if __name__ == "__main__":
    main()