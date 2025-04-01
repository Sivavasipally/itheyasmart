import streamlit as st
import requests
import json
import os
import tempfile
import git
import re
from pathlib import Path
import yaml
from jinja2 import Template
import shutil

# Set page config
st.set_page_config(
    page_title="Spring Boot Code Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 1rem;
}
.section-header {
    font-size: 1.5rem;
    font-weight: bold;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.code-container {
    background-color: #f0f0f0;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ================ TEMPLATES ================

# Java Controller Template
CONTROLLER_TEMPLATE = """package {{ package }}.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import {{ package }}.service.{{ service_name }};
import {{ package }}.model.*;
import java.util.List;
import lombok.extern.slf4j.Slf4j;

/**
 * Controller for {{ api_name }}
 * Generated based on Swagger specification
 */
@RestController
@RequestMapping("{{ base_path }}")
@Slf4j
public class {{ controller_name }} {

    @Autowired
    private {{ service_name }} service;

    {% for endpoint in endpoints %}
    /**
     * {{ endpoint.description }}
     */
    @{{ endpoint.method }}("{{ endpoint.path }}")
    public ResponseEntity<?> {{ endpoint.operation_id }}(
        {% for param in endpoint.parameters %}
        {% if param.in == 'path' %}@PathVariable{% elif param.in == 'query' %}@RequestParam(required={{ param.required|lower }}){% elif param.in == 'body' %}@RequestBody{% endif %} {{ param.type }} {{ param.name }}{% if not loop.last %},
        {% endif %}{% endfor %}
    ) {
        log.info("{{ endpoint.operation_id }} called with parameters: {% for param in endpoint.parameters %}{{ param.name }}={{ '{' }}{{ param.name }}{{ '}' }}{% if not loop.last %}, {% endif %}{% endfor %}");
        {% if endpoint.return_type %}
        {{ endpoint.return_type }} result = service.{{ endpoint.operation_id }}({% for param in endpoint.parameters %}{{ param.name }}{% if not loop.last %}, {% endif %}{% endfor %});
        return ResponseEntity.ok(result);
        {% else %}
        service.{{ endpoint.operation_id }}({% for param in endpoint.parameters %}{{ param.name }}{% if not loop.last %}, {% endif %}{% endfor %});
        return ResponseEntity.ok().build();
        {% endif %}
    }
    {% endfor %}
}
"""

# Java Service Interface Template
SERVICE_INTERFACE_TEMPLATE = """package {{ package }}.service;

import {{ package }}.model.*;
import java.util.List;

/**
 * Service interface for {{ api_name }}
 * Generated based on Swagger specification
 */
public interface {{ service_name }} {
    {% for endpoint in endpoints %}
    /**
     * {{ endpoint.description }}
     */
    {% if endpoint.return_type %}{{ endpoint.return_type }} {% else %}void {% endif %}{{ endpoint.operation_id }}({% for param in endpoint.parameters %}{{ param.type }} {{ param.name }}{% if not loop.last %}, {% endif %}{% endfor %});
    {% endfor %}
}
"""

# Java Service Implementation Template
SERVICE_IMPL_TEMPLATE = """package {{ package }}.service.impl;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import {{ package }}.service.{{ service_name }};
import {{ package }}.client.{{ feign_client_name }};
import {{ package }}.model.*;
import java.util.List;
import lombok.extern.slf4j.Slf4j;

/**
 * Service implementation for {{ api_name }}
 * Generated based on Swagger specification
 */
@Service
@Slf4j
public class {{ service_impl_name }} implements {{ service_name }} {

    @Autowired
    private {{ feign_client_name }} client;

    {% for endpoint in endpoints %}
    @Override
    public {% if endpoint.return_type %}{{ endpoint.return_type }} {% else %}void {% endif %}{{ endpoint.operation_id }}({% for param in endpoint.parameters %}{{ param.type }} {{ param.name }}{% if not loop.last %}, {% endif %}{% endfor %}) {
        log.debug("Executing {{ endpoint.operation_id }} in service");
        {% if endpoint.return_type %}
        return client.{{ endpoint.operation_id }}({% for param in endpoint.parameters %}{{ param.name }}{% if not loop.last %}, {% endif %}{% endfor %});
        {% else %}
        client.{{ endpoint.operation_id }}({% for param in endpoint.parameters %}{{ param.name }}{% if not loop.last %}, {% endif %}{% endfor %});
        {% endif %}
    }
    {% endfor %}
}
"""

# Feign Client Template
FEIGN_CLIENT_TEMPLATE = """package {{ package }}.client;

import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.*;
import {{ package }}.model.*;
import java.util.List;

/**
 * Feign client for {{ api_name }}
 * Generated based on Swagger specification
 */
@FeignClient(name = "{{ service_name|lower }}", url = "${feign.client.{{ service_name|lower }}.url}")
public interface {{ feign_client_name }} {
    {% for endpoint in endpoints %}
    /**
     * {{ endpoint.description }}
     */
    @{{ endpoint.method }}("{{ endpoint.path }}")
    {% if endpoint.return_type %}{{ endpoint.return_type }} {% else %}void {% endif %}{{ endpoint.operation_id }}(
        {% for param in endpoint.parameters %}
        {% if param.in == 'path' %}@PathVariable("{{ param.name }}"){% elif param.in == 'query' %}@RequestParam(value = "{{ param.name }}", required = {{ param.required|lower }}){% elif param.in == 'body' %}@RequestBody{% endif %} {{ param.type }} {{ param.name }}{% if not loop.last %},
        {% endif %}{% endfor %}
    );
    {% endfor %}
}
"""

# Model Template
MODEL_TEMPLATE = """package {{ package }}.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.List;
import java.util.Date;

/**
 * Model class for {{ class_name }}
 * Generated based on Swagger specification
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class {{ class_name }} {
    {% for property in properties %}
    private {{ property.type }} {{ property.name }};
    {% endfor %}
}
"""


# ================ HELPER FUNCTIONS ================

def extract_package_from_git(repo_path):
    """Extract the main package name from the git repository"""
    java_files = list(Path(repo_path).rglob("*.java"))

    if not java_files:
        return "com.example.application"

    for file in java_files:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            package_match = re.search(r'package\s+([\w.]+)', content)
            if package_match:
                package_parts = package_match.group(1).split('.')
                if len(package_parts) >= 2:
                    return '.'.join(package_parts[0:3])

    return "com.example.application"


def convert_swagger_type_to_java(swagger_type, swagger_format=None, ref=None):
    """Convert Swagger types to Java types"""
    if ref:
        # Extract the model name from the reference
        return ref.split('/')[-1]

    type_mapping = {
        'integer': {
            'int32': 'Integer',
            'int64': 'Long',
            None: 'Integer'
        },
        'number': {
            'float': 'Float',
            'double': 'Double',
            None: 'Double'
        },
        'string': {
            'date': 'Date',
            'date-time': 'Date',
            'byte': 'String',
            'binary': 'byte[]',
            'password': 'String',
            None: 'String'
        },
        'boolean': {
            None: 'Boolean'
        },
        'array': {
            None: 'List<?>'
        },
        'object': {
            None: 'Object'
        },
        None: {
            None: 'Object'
        }
    }

    return type_mapping.get(swagger_type, {}).get(swagger_format,
                                                  type_mapping.get(swagger_type, {}).get(None, 'Object'))


def parse_swagger(swagger_data, package_name):
    """Parse Swagger specification to extract API endpoints and models"""
    api_name = swagger_data.get('info', {}).get('title', 'API').replace(' ', '')
    base_path = swagger_data.get('basePath', '')

    if not base_path.startswith('/'):
        base_path = '/' + base_path

    # Extract paths and operations
    endpoints = []
    paths = swagger_data.get('paths', {})

    for path, path_details in paths.items():
        for method, operation in path_details.items():
            if method.lower() not in ['get', 'post', 'put', 'delete', 'patch']:
                continue

            operation_id = operation.get('operationId',
                                         f"{method.lower()}_{path.replace('/', '_').replace('{', '').replace('}', '')}")

            # Extract parameters
            parameters = []
            for param in operation.get('parameters', []):
                param_type = None
                if 'schema' in param:
                    schema = param['schema']
                    if '$ref' in schema:
                        param_type = schema['$ref'].split('/')[-1]
                    else:
                        param_type = convert_swagger_type_to_java(
                            schema.get('type'),
                            schema.get('format'),
                            schema.get('$ref')
                        )
                        if schema.get('type') == 'array' and 'items' in schema:
                            items = schema['items']
                            if '$ref' in items:
                                item_type = items['$ref'].split('/')[-1]
                                param_type = f"List<{item_type}>"
                            else:
                                item_type = convert_swagger_type_to_java(
                                    items.get('type'),
                                    items.get('format')
                                )
                                param_type = f"List<{item_type}>"
                else:
                    param_type = convert_swagger_type_to_java(
                        param.get('type'),
                        param.get('format')
                    )

                parameters.append({
                    'name': param.get('name', ''),
                    'in': param.get('in', ''),
                    'required': param.get('required', False),
                    'type': param_type
                })

            # Determine return type
            return_type = None
            responses = operation.get('responses', {})
            if '200' in responses or '201' in responses:
                success_response = responses.get('200', responses.get('201', {}))
                schema = success_response.get('schema', {})

                if schema:
                    if '$ref' in schema:
                        return_type = schema['$ref'].split('/')[-1]
                    elif schema.get('type') == 'array' and 'items' in schema:
                        items = schema['items']
                        if '$ref' in items:
                            item_type = items['$ref'].split('/')[-1]
                            return_type = f"List<{item_type}>"
                        else:
                            item_type = convert_swagger_type_to_java(
                                items.get('type'),
                                items.get('format')
                            )
                            return_type = f"List<{item_type}>"
                    else:
                        return_type = convert_swagger_type_to_java(
                            schema.get('type'),
                            schema.get('format')
                        )

            endpoints.append({
                'path': path,
                'method': method.upper(),
                'operation_id': operation_id,
                'description': operation.get('summary', 'No description provided'),
                'parameters': parameters,
                'return_type': return_type
            })

    # Extract models
    models = []
    definitions = swagger_data.get('definitions', {})

    for model_name, model_details in definitions.items():
        properties = []
        for prop_name, prop_details in model_details.get('properties', {}).items():
            prop_type = None

            if '$ref' in prop_details:
                prop_type = prop_details['$ref'].split('/')[-1]
            elif prop_details.get('type') == 'array' and 'items' in prop_details:
                items = prop_details['items']
                if '$ref' in items:
                    item_type = items['$ref'].split('/')[-1]
                    prop_type = f"List<{item_type}>"
                else:
                    item_type = convert_swagger_type_to_java(
                        items.get('type'),
                        items.get('format')
                    )
                    prop_type = f"List<{item_type}>"
            else:
                prop_type = convert_swagger_type_to_java(
                    prop_details.get('type'),
                    prop_details.get('format')
                )

            properties.append({
                'name': prop_name,
                'type': prop_type
            })

        models.append({
            'class_name': model_name,
            'properties': properties
        })

    return {
        'api_name': api_name,
        'base_path': base_path,
        'endpoints': endpoints,
        'models': models,
        'package': package_name
    }


def generate_controller(api_data):
    """Generate controller class"""
    controller_name = f"{api_data['api_name']}Controller"
    service_name = f"{api_data['api_name']}Service"

    template = Template(CONTROLLER_TEMPLATE)
    return template.render(
        controller_name=controller_name,
        service_name=service_name,
        api_name=api_data['api_name'],
        base_path=api_data['base_path'],
        endpoints=api_data['endpoints'],
        package=api_data['package']
    )


def generate_service_interface(api_data):
    """Generate service interface"""
    service_name = f"{api_data['api_name']}Service"

    template = Template(SERVICE_INTERFACE_TEMPLATE)
    return template.render(
        service_name=service_name,
        api_name=api_data['api_name'],
        endpoints=api_data['endpoints'],
        package=api_data['package']
    )


def generate_service_impl(api_data):
    """Generate service implementation"""
    service_name = f"{api_data['api_name']}Service"
    service_impl_name = f"{api_data['api_name']}ServiceImpl"
    feign_client_name = f"{api_data['api_name']}Client"

    template = Template(SERVICE_IMPL_TEMPLATE)
    return template.render(
        service_name=service_name,
        service_impl_name=service_impl_name,
        feign_client_name=feign_client_name,
        api_name=api_data['api_name'],
        endpoints=api_data['endpoints'],
        package=api_data['package']
    )


def generate_feign_client(api_data):
    """Generate Feign client"""
    feign_client_name = f"{api_data['api_name']}Client"
    service_name = f"{api_data['api_name']}Service"

    template = Template(FEIGN_CLIENT_TEMPLATE)
    return template.render(
        feign_client_name=feign_client_name,
        service_name=service_name,
        api_name=api_data['api_name'],
        endpoints=api_data['endpoints'],
        package=api_data['package']
    )


def generate_model(model_data, package):
    """Generate model class"""
    template = Template(MODEL_TEMPLATE)
    return template.render(
        class_name=model_data['class_name'],
        properties=model_data['properties'],
        package=package
    )


def create_project_structure(output_dir, api_data):
    """Create project structure and write generated files"""
    # Create package directories
    package_path = api_data['package'].replace('.', '/')
    controller_dir = os.path.join(output_dir, 'src/main/java', package_path, 'controller')
    service_dir = os.path.join(output_dir, 'src/main/java', package_path, 'service')
    service_impl_dir = os.path.join(output_dir, 'src/main/java', package_path, 'service/impl')
    client_dir = os.path.join(output_dir, 'src/main/java', package_path, 'client')
    model_dir = os.path.join(output_dir, 'src/main/java', package_path, 'model')

    os.makedirs(controller_dir, exist_ok=True)
    os.makedirs(service_dir, exist_ok=True)
    os.makedirs(service_impl_dir, exist_ok=True)
    os.makedirs(client_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Generate controller
    controller_name = f"{api_data['api_name']}Controller"
    controller_content = generate_controller(api_data)
    with open(os.path.join(controller_dir, f"{controller_name}.java"), 'w') as f:
        f.write(controller_content)

    # Generate service interface
    service_name = f"{api_data['api_name']}Service"
    service_content = generate_service_interface(api_data)
    with open(os.path.join(service_dir, f"{service_name}.java"), 'w') as f:
        f.write(service_content)

    # Generate service implementation
    service_impl_name = f"{api_data['api_name']}ServiceImpl"
    service_impl_content = generate_service_impl(api_data)
    with open(os.path.join(service_impl_dir, f"{service_impl_name}.java"), 'w') as f:
        f.write(service_impl_content)

    # Generate Feign client
    feign_client_name = f"{api_data['api_name']}Client"
    feign_client_content = generate_feign_client(api_data)
    with open(os.path.join(client_dir, f"{feign_client_name}.java"), 'w') as f:
        f.write(feign_client_content)

    # Generate models
    for model in api_data['models']:
        model_content = generate_model(model, api_data['package'])
        with open(os.path.join(model_dir, f"{model['class_name']}.java"), 'w') as f:
            f.write(model_content)

    # Create application.properties with Feign client configuration
    properties_dir = os.path.join(output_dir, 'src/main/resources')
    os.makedirs(properties_dir, exist_ok=True)

    with open(os.path.join(properties_dir, 'application.properties'), 'w') as f:
        f.write(f"# Feign client configuration\n")
        f.write(f"feign.client.{service_name.lower()}.url=http://example-service-url\n")

    # Create pom.xml with required dependencies
    with open(os.path.join(output_dir, 'pom.xml'), 'w') as f:
        f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0" 
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>2.7.0</version>
        <relativePath/>
    </parent>

    <groupId>{api_data['package'].split('.')[0]}.{api_data['package'].split('.')[1]}</groupId>
    <artifactId>{api_data['api_name'].lower()}-service</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <name>{api_data['api_name']} Service</name>
    <description>Experience API for {api_data['api_name']}</description>

    <properties>
        <java.version>11</java.version>
        <spring-cloud.version>2021.0.3</spring-cloud.version>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.cloud</groupId>
            <artifactId>spring-cloud-starter-openfeign</artifactId>
        </dependency>
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <optional>true</optional>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>org.springframework.cloud</groupId>
                <artifactId>spring-cloud-dependencies</artifactId>
                <version>springcloud.version</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>
        </dependencies>
    </dependencyManagement>

    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
                <configuration>
                    <excludes>
                        <exclude>
                            <groupId>org.projectlombok</groupId>
                            <artifactId>lombok</artifactId>
                        </exclude>
                    </excludes>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
""")

    # Create main application class
    main_class_dir = os.path.join(output_dir, 'src/main/java', package_path)
    os.makedirs(main_class_dir, exist_ok=True)

    main_class_name = f"{api_data['api_name']}Application"
    with open(os.path.join(main_class_dir, f"{main_class_name}.java"), 'w') as f:
        f.write(f"""package {api_data['package']};

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.openfeign.EnableFeignClients;

@SpringBootApplication
@EnableFeignClients
public class {main_class_name} {{
    public static void main(String[] args) {{
        SpringApplication.run({main_class_name}.class, args);
    }}
}}
""")

    return output_dir


# ================ UI COMPONENTS ================

st.markdown('<div class="main-header">Spring Boot Code Generator</div>', unsafe_allow_html=True)
st.write("Generate Spring Boot microservice code with Experience APIs and Feign clients from Swagger specifications")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        git_repo = st.text_input("Git Repository URL", placeholder="https://github.com/username/repo.git")
        git_branch = st.text_input("Git Branch (optional)", "main")

    with col2:
        swagger_input_type = st.radio("Swagger Input Type", ["URL", "JSON/YAML Text"])

        if swagger_input_type == "URL":
            swagger_url = st.text_input("Swagger URL", placeholder="https://example.com/swagger.json")
            swagger_text = None
        else:
            swagger_text = st.text_area("Swagger JSON/YAML",
                                        placeholder='{\n  "swagger": "2.0",\n  "info": {\n    "title": "Example API"\n  }\n}')
            swagger_url = None

    submit_button = st.form_submit_button("Generate Code")

if submit_button:
    with st.spinner("Generating Spring Boot code..."):
        # Create a temporary directory for git clone
        temp_dir = tempfile.mkdtemp()
        output_dir = tempfile.mkdtemp()

        try:
            # Clone the git repository
            if git_repo:
                try:
                    repo = git.Repo.clone_from(git_repo, temp_dir, branch=git_branch)
                    st.success(f"Successfully cloned repository: {git_repo}")

                    # Extract package name from git repo
                    package_name = extract_package_from_git(temp_dir)
                    st.info(f"Detected package name: {package_name}")
                except Exception as e:
                    st.error(f"Error cloning repository: {str(e)}")
                    st.warning("Proceeding with default package name: com.example.application")
                    package_name = "com.example.application"
            else:
                st.warning("No Git repository provided. Using default package name: com.example.application")
                package_name = "com.example.application"

            # Parse Swagger
            swagger_data = None

            if swagger_input_type == "URL" and swagger_url:
                try:
                    response = requests.get(swagger_url)
                    response.raise_for_status()
                    content_type = response.headers.get('Content-Type', '')

                    if 'json' in content_type:
                        swagger_data = response.json()
                    elif 'yaml' in content_type or 'yml' in content_type:
                        swagger_data = yaml.safe_load(response.text)
                    else:
                        # Try to guess format
                        try:
                            swagger_data = response.json()
                        except:
                            try:
                                swagger_data = yaml.safe_load(response.text)
                            except:
                                st.error("Could not parse Swagger specification from URL")

                    st.success(f"Successfully loaded Swagger from URL: {swagger_url}")
                except Exception as e:
                    st.error(f"Error loading Swagger from URL: {str(e)}")
            elif swagger_input_type == "JSON/YAML Text" and swagger_text:
                try:
                    # Try to parse as JSON first
                    try:
                        swagger_data = json.loads(swagger_text)
                    except:
                        # Try YAML if JSON fails
                        swagger_data = yaml.safe_load(swagger_text)

                    st.success("Successfully parsed Swagger from text input")
                except Exception as e:
                    st.error(f"Error parsing Swagger text: {str(e)}")

            if swagger_data:
                # Parse and generate code
                api_data = parse_swagger(swagger_data, package_name)

                # Create project structure
                project_dir = create_project_structure(output_dir, api_data)

                # Create zip file for download
                zip_path = os.path.join(tempfile.gettempdir(), f"{api_data['api_name']}-service.zip")
                shutil.make_archive(zip_path[:-4], 'zip', output_dir)

                # Read the zip file
                with open(zip_path, 'rb') as f:
                    zip_data = f.read()

                # Offer download
                st.success("Code generation completed successfully!")
                st.download_button(
                    label="Download Generated Code",
                    data=zip_data,
                    file_name=f"{api_data['api_name']}-service.zip",
                    mime="application/zip"
                )

                # Show generated files
                st.markdown('<div class="section-header">Generated Files</div>', unsafe_allow_html=True)

                tabs = st.tabs([
                    "Controller",
                    "Service Interface",
                    "Service Implementation",
                    "Feign Client",
                    "Models",
                    "Project Structure"
                ])

                with tabs[0]:
                    st.markdown('<div class="code-container">', unsafe_allow_html=True)
                    st.code(generate_controller(api_data), language="java")
                    st.markdown('</div>', unsafe_allow_html=True)

                with tabs[1]:
                    st.markdown('<div class="code-container">', unsafe_allow_html=True)
                    st.code(generate_service_interface(api_data), language="java")
                    st.markdown('</div>', unsafe_allow_html=True)

                with tabs[2]:
                    st.markdown('<div class="code-container">', unsafe_allow_html=True)
                    st.code(generate_service_impl(api_data), language="java")
                    st.markdown('</div>', unsafe_allow_html=True)

                with tabs[3]:
                    st.markdown('<div class="code-container">', unsafe_allow_html=True)
                    st.code(generate_feign_client(api_data), language="java")
                    st.markdown('</div>', unsafe_allow_html=True)

                with tabs[4]:
                    for model in api_data['models']:
                        st.markdown(f"<b>{model['class_name']}.java</b>", unsafe_allow_html=True)
                        st.markdown('<div class="code-container">', unsafe_allow_html=True)
                        st.code(generate_model(model, api_data['package']), language="java")
                        st.markdown('</div>', unsafe_allow_html=True)

                        with tabs[5]:
                            st.markdown("### Project Structure")
                            st.markdown(f"""
                        ```
                        {api_data['api_name']}-service/
                        ├── pom.xml
                        ├── src/
                        │   ├── main/
                        │   │   ├── java/
                        │   │   │   └── {api_data['package'].replace('.', '/')}/
                        │   │   │       ├── {api_data['api_name']}Application.java
                        │   │   │       ├── controller/
                        │   │   │       │   └── {api_data['api_name']}Controller.java
                        │   │   │       ├── service/
                        │   │   │       │   ├── {api_data['api_name']}Service.java
                        │   │   │       │   └── impl/
                        │   │   │       │       └── {api_data['api_name']}ServiceImpl.java
                        │   │   │       ├── client/
                        │   │   │       │   └── {api_data['api_name']}Client.java
                        │   │   │       └── model/
                        """)
                            for model in api_data['models']:
                                st.markdown(f"│   │   │       │   └── {model['class_name']}.java")

                            st.markdown(f"""│   │   └── resources/
                        │   │       └── application.properties
                        │   └── test/
                        │       └── java/
                        └── target/
                        ```
                        """)
                    else:
                        st.error("No valid Swagger specification provided. Please check your input.")

        except Exception as e:
            st.error(f"Error during code generation: {str(e)}")

        finally:
            #Clean up temporary directories
            try:
                shutil.rmtree(temp_dir)
                shutil.rmtree(output_dir)
            except:
                pass

            # ================ HELP SECTION ================

            with st.expander("How to Use This Tool"):
                st.markdown("""
                            ### How to Use the Spring Boot Code Generator

                            This tool generates a complete Spring Boot microservice with Experience APIs and Feign clients from a Swagger specification. Here's how to use it:

                            1. **Git Repository Input**
                               - Enter a Git repository URL to extract context like package names
                               - You can optionally specify a branch (defaults to 'main')
                               - If no Git repository is provided, default package 'com.example.application' will be used

                            2. **Swagger Specification Input**
                               - You can provide the Swagger spec as either a URL or direct JSON/YAML text
                               - The tool supports both Swagger 2.0 and OpenAPI 3.0 formats

                            3. **Code Generation**
                               - Click "Generate Code" to create the Spring Boot project
                               - The tool will analyze the Swagger spec and Git repo to produce custom code
                               - Download the generated code as a ZIP file

                            ### Generated Code Structure

                            The tool generates a complete Spring Boot project with:

                            - **Controller Layer**: Handles HTTP requests as defined in the Swagger
                            - **Service Layer**: Interface and implementation to process business logic
                            - **Feign Client**: For making HTTP calls to backend services
                            - **Model Classes**: Based on the Swagger definitions
                            - **Maven Configuration**: pom.xml with all necessary dependencies
                            - **Application Properties**: With Feign client configuration

                            ### Requirements for Your Swagger

                            For best results, your Swagger should include:
                            - Defined models/schemas in the 'definitions' or 'components/schemas' section
                            - Operation IDs for each endpoint
                            - Clear path parameters, query parameters, and request/response bodies
                            """)

            with st.expander("Example Swagger"):
                st.code("""
                        {
                          "swagger": "2.0",
                          "info": {
                            "title": "Customer API",
                            "description": "API for customer management",
                            "version": "1.0.0"
                          },
                          "basePath": "/api/v1",
                          "paths": {
                            "/customers": {
                              "get": {
                                "summary": "Get all customers",
                                "operationId": "getAllCustomers",
                                "responses": {
                                  "200": {
                                    "description": "Success",
                                    "schema": {
                                      "type": "array",
                                      "items": {
                                        "$ref": "#/definitions/Customer"
                                      }
                                    }
                                  }
                                }
                              },
                              "post": {
                                "summary": "Create new customer",
                                "operationId": "createCustomer",
                                "parameters": [
                                  {
                                    "in": "body",
                                    "name": "customer",
                                    "required": true,
                                    "schema": {
                                      "$ref": "#/definitions/Customer"
                                    }
                                  }
                                ],
                                "responses": {
                                  "201": {
                                    "description": "Created",
                                    "schema": {
                                      "$ref": "#/definitions/Customer"
                                    }
                                  }
                                }
                              }
                            },
                            "/customers/{id}": {
                              "get": {
                                "summary": "Get customer by ID",
                                "operationId": "getCustomerById",
                                "parameters": [
                                  {
                                    "in": "path",
                                    "name": "id",
                                    "required": true,
                                    "type": "string"
                                  }
                                ],
                                "responses": {
                                  "200": {
                                    "description": "Success",
                                    "schema": {
                                      "$ref": "#/definitions/Customer"
                                    }
                                  }
                                }
                              },
                              "put": {
                                "summary": "Update customer",
                                "operationId": "updateCustomer",
                                "parameters": [
                                  {
                                    "in": "path",
                                    "name": "id",
                                    "required": true,
                                    "type": "string"
                                  },
                                  {
                                    "in": "body",
                                    "name": "customer",
                                    "required": true,
                                    "schema": {
                                      "$ref": "#/definitions/Customer"
                                    }
                                  }
                                ],
                                "responses": {
                                  "200": {
                                    "description": "Success",
                                    "schema": {
                                      "$ref": "#/definitions/Customer"
                                    }
                                  }
                                }
                              },
                              "delete": {
                                "summary": "Delete customer",
                                "operationId": "deleteCustomer",
                                "parameters": [
                                  {
                                    "in": "path",
                                    "name": "id",
                                    "required": true,
                                    "type": "string"
                                  }
                                ],
                                "responses": {
                                  "204": {
                                    "description": "No Content"
                                  }
                                }
                              }
                            }
                          },
                          "definitions": {
                            "Customer": {
                              "type": "object",
                              "properties": {
                                "id": {
                                  "type": "string"
                                },
                                "firstName": {
                                  "type": "string"
                                },
                                "lastName": {
                                  "type": "string"
                                },
                                "email": {
                                  "type": "string"
                                },
                                "phone": {
                                  "type": "string"
                                },
                                "address": {
                                  "$ref": "#/definitions/Address"
                                },
                                "createdAt": {
                                  "type": "string",
                                  "format": "date-time"
                                }
                              }
                            },
                            "Address": {
                              "type": "object",
                              "properties": {
                                "street": {
                                  "type": "string"
                                },
                                "city": {
                                  "type": "string"
                                },
                                "state": {
                                  "type": "string"
                                },
                                "zipCode": {
                                  "type": "string"
                                },
                                "country": {
                                  "type": "string"
                                }
                              }
                            }
                          }
                        }
                        """, language="json")