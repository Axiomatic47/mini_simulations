"""
Equation Coverage Analyzer
This tool analyzes your equation sets for completeness, consistency, and optimization opportunities.
"""

import inspect
import importlib
import logging
import numpy as np
import matplotlib.pyplot as plt
import physics_domains.strong_nuclear.identity_binding
import physics_domains.strong_nuclear.civilization_oscillation
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EquationAnalyzer")


class EquationCoverageAnalyzer:
    """
    Analyzes the coverage and completeness of equation sets across different scales and physics domains.
    """

    def __init__(self):
        """Initialize the equation coverage analyzer."""
        self.physics_domains = [
            "thermodynamics",
            "electromagnetism",
            "strong_nuclear",
            "weak_nuclear",
            "quantum_mechanics",
            "relativity",
            "astrophysics",
            "multi_system"
        ]

        self.application_domains = [
            "intelligence",
            "knowledge",
            "truth",
            "wisdom",
            "suppression",
            "resistance",
            "free_will",
            "civilization"
        ]

        self.scale_levels = [
            "quantum",
            "agent",
            "group",
            "civilization",
            "multi_civilization",
            "cosmic"
        ]

        # Define expected equations by domain
        self.expected_equations = {
            "thermodynamics": ["intelligence_growth", "entropy_resistance"],
            "electromagnetism": ["wisdom_field", "knowledge_field_influence", "free_will_decision"],
            "strong_nuclear": ["identity_binding", "civilization_oscillation"],
            "weak_nuclear": ["suppression_feedback", "knowledge_growth_phase_transition"],
            "quantum_mechanics": ["quantum_tunneling_probability", "quantum_entanglement_correlation"],
            "relativity": ["truth_adoption"],
            "astrophysics": ["civilization_lifecycle_phase", "suppression_event_horizon",
                             "cosmic_background_knowledge", "knowledge_inflation",
                             "knowledge_gravitational_lensing", "dark_energy_knowledge_acceleration",
                             "galactic_structure_model"],
            "multi_system": ["knowledge_diffusion", "cultural_influence", "resource_competition",
                             "civilization_movement", "calculate_interaction_strength"]
        }

    def analyze_equation_set(self, equation_modules=None, identify_gaps=True):
        """
        Analyze the coverage of equations across physics domains and application areas.

        Args:
            equation_modules: List of module names containing equations
            identify_gaps: Whether to identify gaps in equation coverage

        Returns:
            Dictionary containing analysis results
        """
        if equation_modules is None:
            equation_modules = ["equations", "astrophysics_extensions",
                                "quantum_em_extensions", "multi_civilization_extensions"]

        logger.info(f"Analyzing equation coverage for modules: {equation_modules}")

        # Load modules and extract functions
        equations = self._load_equations(equation_modules)

        # Analyze coverage
        coverage = self._analyze_coverage(equations)

        # Identify gaps if requested
        gaps = []
        if identify_gaps:
            gaps = self._identify_gaps(coverage)

        # Analyze cross-domain integration
        integration = self._analyze_integration(equations)

        return {
            "equations": equations,
            "coverage": coverage,
            "gaps": gaps,
            "integration": integration
        }

    def _load_equations(self, module_names):
        """
        Load equation functions from specified modules.

        Args:
            module_names: List of module names to load

        Returns:
            Dictionary of equations by module and function name
        """
        equations = {}

        for module_name in module_names:
            try:
                # Try different import strategies
                module = None
                import_errors = []

                # Strategy 1: Try importing the module directly
                try:
                    module = importlib.import_module(module_name)
                    logger.info(f"Successfully imported {module_name} directly")
                except ImportError as e:
                    import_errors.append(f"Direct import failed: {str(e)}")

                # Strategy 2: Try with config prefix if it's not already there
                if module is None and not module_name.startswith('config.'):
                    try:
                        full_module_name = f"config.{module_name}"
                        module = importlib.import_module(full_module_name)
                        logger.info(f"Successfully imported {module_name} as {full_module_name}")
                    except ImportError as e:
                        import_errors.append(f"Config prefix import failed: {str(e)}")

                # Strategy 3: Special handling for specific modules
                if module is None:
                    # For physics_domains.strong_nuclear
                    if module_name == 'physics_domains.strong_nuclear':
                        try:
                            # Try to import identity_binding and civilization_oscillation directly
                            identity_binding_module = importlib.import_module(
                                'physics_domains.strong_nuclear.identity_binding')
                            civilization_oscillation_module = importlib.import_module(
                                'physics_domains.strong_nuclear.civilization_oscillation')

                            # Create a composite module
                            import types
                            module = types.ModuleType('physics_domains.strong_nuclear')

                            # Copy functions from the individual modules
                            if hasattr(identity_binding_module, 'identity_binding'):
                                setattr(module, 'identity_binding', identity_binding_module.identity_binding)

                            if hasattr(civilization_oscillation_module, 'civilization_oscillation'):
                                setattr(module, 'civilization_oscillation',
                                        civilization_oscillation_module.civilization_oscillation)

                            logger.info(f"Successfully created composite module for {module_name}")
                        except ImportError as e:
                            import_errors.append(f"Strong nuclear special handling failed: {str(e)}")

                    # For bridge_functions
                    elif module_name == 'bridge_functions':
                        try:
                            # Try importing from the root directory
                            module = importlib.import_module('bridge_functions')
                            logger.info(f"Successfully imported {module_name} from root")
                        except ImportError as e:
                            import_errors.append(f"Bridge functions root import failed: {str(e)}")

                # If all strategies failed, log the errors and continue to the next module
                if module is None:
                    error_msg = f"Could not import module {module_name}: {'; '.join(import_errors)}"
                    logger.error(error_msg)
                    continue

                # Get all functions from the module
                module_functions = {}
                for name, func in inspect.getmembers(module, inspect.isfunction):
                    # Skip private functions
                    if name.startswith('_'):
                        continue

                    # Extract function information
                    try:
                        source = inspect.getsource(func)
                        signature = str(inspect.signature(func))
                        docstring = inspect.getdoc(func) or ""

                        # Determine physics domain
                        domain = self._determine_physics_domain(name, docstring, source)

                        # Determine application domain
                        application = self._determine_application_domain(name, docstring, source)

                        # Determine scale level
                        scale = self._determine_scale_level(name, docstring, source)

                        module_functions[name] = {
                            "name": name,
                            "signature": signature,
                            "docstring": docstring,
                            "source_length": len(source),
                            "physics_domain": domain,
                            "application_domain": application,
                            "scale_level": scale
                        }
                    except Exception as e:
                        logger.warning(f"Error processing function {name}: {str(e)}")
                        continue

                equations[module_name] = module_functions
                logger.info(f"Loaded {len(module_functions)} equations from {module_name}")

            except Exception as e:
                logger.error(f"Error processing module {module_name}: {str(e)}")

        return equations

    def scan_equations(self, domains=None):
        """
        Scan through modules to find and categorize equation functions.

        Args:
            domains: List of module names to scan or None to scan all

        Returns:
            Dictionary of categorized equations
        """
        import inspect
        import importlib
        import re

        if domains is None:
            # Scan both config and physics_domains directories
            domains = [
                # Config modules
                "config.equations",
                "config.astrophysics_extensions",
                "config.quantum_em_extensions",
                "config.multi_civilization_extensions",

                # Physics domains
                "physics_domains.thermodynamics",
                "physics_domains.relativity",
                "physics_domains.electromagnetism",
                "physics_domains.quantum_mechanics",
                "physics_domains.weak_nuclear",
                "physics_domains.strong_nuclear",
                "physics_domains.astrophysics",
                "physics_domains.multi_system"
            ]

        categorized_equations = {
            "by_physics_domain": {},
            "by_scale_level": {},
            "by_application_domain": {},
            "functions": {}
        }

        # Initialize known categories
        for domain in self.known_physics_domains:
            categorized_equations["by_physics_domain"][domain] = []

        for level in self.known_scale_levels:
            categorized_equations["by_scale_level"][level] = []

        for domain in self.known_application_domains:
            categorized_equations["by_application_domain"][domain] = []

        # Regular expressions for metadata extraction
        physics_domain_pattern = r"Physics Domain:\s*(\w+)"
        scale_level_pattern = r"Scale Level:\s*(\w+)"
        application_domains_pattern = r"Application Domains:\s*([\w\s,]+)"

        # Scan each module
        for module_name in domains:
            try:
                # First try direct import
                try:
                    module = importlib.import_module(module_name)
                except ImportError:
                    # If that fails, try recursive import for physics_domains
                    if "physics_domains" in module_name:
                        base_domain = module_name.split(".")[-1]
                        for func_name in [
                            "intelligence_growth", "truth_adoption", "wisdom_field",
                            "suppression_feedback", "resistance_resurgence", "civilization_oscillation",
                            "identity_binding", "quantum_tunneling_probability", "quantum_entanglement_correlation",
                            "build_entanglement_network", "knowledge_field_gradient", "knowledge_field_influence",
                            "free_will_decision", "knowledge_growth_phase_transition",
                            "dark_energy_knowledge_acceleration",
                            "cosmic_background_knowledge", "galactic_structure_model", "civilization_lifecycle_phase",
                            "suppression_event_horizon", "knowledge_gravitational_lensing", "knowledge_inflation",
                            "calculate_distance_matrix", "calculate_interaction_strength", "civilization_movement",
                            "cultural_influence", "detect_civilization_collapse", "detect_civilization_mergers",
                            "galactic_collision_effect", "initialize_civilizations", "knowledge_diffusion",
                            "process_all_civilization_interactions", "process_civilization_merger",
                            "remove_civilization",
                            "resource_competition", "spawn_new_civilization", "update_civilization_sizes"
                        ]:
                            try:
                                # Try to import specific function modules from physics_domains
                                specific_module = importlib.import_module(f"physics_domains.{base_domain}.{func_name}")
                                if hasattr(specific_module, func_name):
                                    func = getattr(specific_module, func_name)
                                    self._process_function(func, func_name,
                                                           f"physics_domains.{base_domain}.{func_name}",
                                                           categorized_equations, physics_domain_pattern,
                                                           scale_level_pattern, application_domains_pattern)
                            except ImportError:
                                continue
                    continue

                # Get functions from the module
                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj) and not name.startswith("_"):
                        self._process_function(obj, name, module_name, categorized_equations,
                                               physics_domain_pattern, scale_level_pattern,
                                               application_domains_pattern)

            except ImportError as e:
                self.logger.warning(f"Could not import module {module_name}: {e}")
            except Exception as e:
                self.logger.error(f"Error scanning module {module_name}: {e}")

        return categorized_equations

    def _process_function(self, func, func_name, module_name, categorized_equations,
                          physics_domain_pattern, scale_level_pattern, application_domains_pattern):
        """Process a single function and categorize it based on docstring."""
        docstring = inspect.getdoc(func)
        if not docstring:
            return

        # Extract metadata from docstring
        physics_domain = None
        scale_level = None
        application_domains = []

        # Extract physics domain
        match = re.search(physics_domain_pattern, docstring, re.IGNORECASE)
        if match:
            physics_domain = match.group(1).lower()
            if physics_domain in self.known_physics_domains:
                categorized_equations["by_physics_domain"][physics_domain].append(func_name)

        # Extract scale level
        match = re.search(scale_level_pattern, docstring, re.IGNORECASE)
        if match:
            scale_level = match.group(1).lower()
            if scale_level in self.known_scale_levels:
                categorized_equations["by_scale_level"][scale_level].append(func_name)

        # Extract application domains
        match = re.search(application_domains_pattern, docstring, re.IGNORECASE)
        if match:
            domains_str = match.group(1)
            app_domains = [d.strip().lower() for d in domains_str.split(",")]
            for domain in app_domains:
                if domain in self.known_application_domains:
                    categorized_equations["by_application_domain"][domain].append(func_name)
            application_domains = app_domains

        # Store function metadata
        categorized_equations["functions"][func_name] = {
            "name": func_name,
            "module": module_name,
            "physics_domain": physics_domain,
            "scale_level": scale_level,
            "application_domains": application_domains,
            "docstring": docstring,
            "signature": str(inspect.signature(func))
        }

    def _determine_physics_domain(self, name, docstring, source):
        """
        Determine the physics domain for an equation based on name, docstring, and source.

        Args:
            name: Function name
            docstring: Function docstring
            source: Function source code

        Returns:
            String representing the physics domain
        """
        # First, look for explicit Physics Domain annotation in docstring
        if docstring:
            for line in docstring.split('\n'):
                if line.strip().startswith('Physics Domain:'):
                    domain = line.strip()[len('Physics Domain:'):].strip().lower()
                    if domain in self.physics_domains:
                        return domain

        # Look for keywords in name, docstring, and source
        combined_text = f"{name} {docstring} {source}".lower()

        domain_keywords = {
            "thermodynamics": ["thermodynamic", "entropy", "energy", "heat", "intelligence_growth"],
            "electromagnetism": ["electromagnetic", "electric", "field", "charge", "wisdom_field", "field_influence"],
            "strong_nuclear": ["strong nuclear", "binding", "identity", "oscillation"],
            "weak_nuclear": ["weak nuclear", "decay", "transformation", "phase transition", "resurgence"],
            "quantum_mechanics": ["quantum", "tunneling", "entanglement", "uncertainty", "superposition"],
            "relativity": ["relativ", "speed limit", "time dilation", "truth_adoption"],
            "astrophysics": ["astrophysic", "stellar", "cosmic", "gravity", "lifecycle", "event horizon"],
            "multi_system": ["multi", "interaction", "civilization", "diffusion", "cultural"]
        }

        # Direct mappings based on function names
        name_mappings = {
            "intelligence_growth": "thermodynamics",
            "free_will_decision": "electromagnetism",
            "truth_adoption": "relativity",
            "wisdom_field": "electromagnetism",
            "resistance_resurgence": "weak_nuclear",
            "suppression_feedback": "weak_nuclear",
            "quantum_tunneling_probability": "quantum_mechanics",
            "knowledge_field_influence": "electromagnetism",
            "civilization_lifecycle_phase": "astrophysics",
            "suppression_event_horizon": "astrophysics",
            "knowledge_gravitational_lensing": "astrophysics",
            "knowledge_inflation": "astrophysics",
            "cosmic_background_knowledge": "astrophysics",
            "dark_energy_knowledge_acceleration": "astrophysics",
            "galactic_structure_model": "astrophysics",
            "knowledge_diffusion": "multi_system",
            "cultural_influence": "multi_system",
            "resource_competition": "multi_system",
            "civilization_movement": "multi_system",
            "identity_binding": "strong_nuclear",
            "civilization_oscillation": "strong_nuclear"
        }

        # Check direct mapping first
        if name in name_mappings:
            return name_mappings[name]

        # Otherwise, look for keywords
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in combined_text:
                    return domain

        # Default if no match found
        return "unknown"

    def _determine_application_domain(self, name, docstring, source):
        """
        Determine the application domain for an equation.

        Args:
            name: Function name
            docstring: Function docstring
            source: Function source code

        Returns:
            String representing the application domain
        """
        # First, look for explicit Application Domains annotation in docstring
        if docstring:
            for line in docstring.split('\n'):
                if line.strip().startswith('Application Domains:'):
                    domains_str = line.strip()[len('Application Domains:'):].strip().lower()
                    domains = [d.strip() for d in domains_str.split(',')]
                    for domain in domains:
                        if domain in self.application_domains:
                            return domain
                    # If we got domains but none match our known list, return the first one
                    if domains:
                        return domains[0]

        combined_text = f"{name} {docstring} {source}".lower()

        domain_keywords = {
            "intelligence": ["intelligence", "growth", "learning"],
            "knowledge": ["knowledge", "info", "understanding"],
            "truth": ["truth", "adoption", "awareness"],
            "wisdom": ["wisdom", "integration", "understanding"],
            "suppression": ["suppression", "limitation", "control"],
            "resistance": ["resistance", "opposition", "barrier"],
            "free_will": ["free will", "decision", "choice"],
            "civilization": ["civilization", "society", "social"]
        }

        # Name-based mappings
        name_mappings = {
            "intelligence_growth": "intelligence",
            "free_will_decision": "free_will",
            "truth_adoption": "truth",
            "wisdom_field": "wisdom",
            "resistance_resurgence": "resistance",
            "suppression_feedback": "suppression",
            "quantum_tunneling_probability": "knowledge",
            "knowledge_field_influence": "knowledge",
            "civilization_lifecycle_phase": "civilization",
            "suppression_event_horizon": "suppression",
            "knowledge_gravitational_lensing": "knowledge",
            "cosmic_background_knowledge": "knowledge",
            "galactic_structure_model": "civilization",
            "identity_binding": "knowledge",
            "civilization_oscillation": "civilization"
        }

        # Check direct mapping first
        if name in name_mappings:
            return name_mappings[name]

        # Otherwise, look for keywords
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in combined_text:
                    return domain

        return "unknown"

    def _determine_scale_level(self, name, docstring, source):
        """
        Determine the scale level for an equation.

        Args:
            name: Function name
            docstring: Function docstring
            source: Function source code

        Returns:
            String representing the scale level
        """
        # First, look for explicit Scale Level annotation in docstring
        if docstring:
            for line in docstring.split('\n'):
                if line.strip().startswith('Scale Level:'):
                    scale = line.strip()[len('Scale Level:'):].strip().lower()
                    if scale in self.scale_levels:
                        return scale
                # Also check for Scale Levels (plural)
                elif line.strip().startswith('Scale Levels:'):
                    scales_str = line.strip()[len('Scale Levels:'):].strip().lower()
                    scales = [s.strip() for s in scales_str.split(',')]
                    for scale in scales:
                        if scale in self.scale_levels:
                            return scale
                    # If we got scales but none match our known list, return the first one
                    if scales:
                        return scales[0]

        combined_text = f"{name} {docstring} {source}".lower()

        scale_keywords = {
            "quantum": ["quantum", "tunneling", "entanglement"],
            "agent": ["agent", "individual", "person", "free will"],
            "group": ["group", "network", "multiple agents"],
            "civilization": ["civilization", "society", "culture"],
            "multi_civilization": ["multi", "interaction", "diffusion"],
            "cosmic": ["cosmic", "universal", "galactic", "lifecycle"]
        }

        # Name-based mappings
        name_mappings = {
            "intelligence_growth": "agent",
            "free_will_decision": "agent",
            "truth_adoption": "agent",
            "wisdom_field": "agent",
            "resistance_resurgence": "agent",
            "suppression_feedback": "agent",
            "quantum_tunneling_probability": "quantum",
            "quantum_entanglement_correlation": "quantum",
            "knowledge_field_influence": "group",
            "knowledge_field_gradient": "group",
            "build_entanglement_network": "group",
            "civilization_lifecycle_phase": "civilization",
            "suppression_event_horizon": "civilization",
            "knowledge_gravitational_lensing": "civilization",
            "knowledge_diffusion": "multi_civilization",
            "cultural_influence": "multi_civilization",
            "resource_competition": "multi_civilization",
            "civilization_movement": "multi_civilization",
            "galactic_structure_model": "cosmic",
            "cosmic_background_knowledge": "cosmic",
            "dark_energy_knowledge_acceleration": "cosmic",
            "identity_binding": "agent",
            "civilization_oscillation": "civilization"
        }

        # Check direct mapping first
        if name in name_mappings:
            return name_mappings[name]

        # Otherwise, look for keywords
        for scale, keywords in scale_keywords.items():
            for keyword in keywords:
                if keyword in combined_text:
                    return scale

        return "unknown"

    def _analyze_coverage(self, equations):
        """
        Analyze the coverage of equations across domains and scales.

        Args:
            equations: Dictionary of equations

        Returns:
            Dictionary containing coverage metrics
        """
        # Initialize coverage metrics
        coverage = {
            "physics_domains": {domain: 0 for domain in self.physics_domains},
            "application_domains": {domain: 0 for domain in self.application_domains},
            "scale_levels": {level: 0 for level in self.scale_levels},
            "physics_by_module": {},
            "scale_by_module": {},
            "cross_domain_coverage": np.zeros((len(self.physics_domains), len(self.application_domains))),
            "cross_scale_coverage": np.zeros((len(self.physics_domains), len(self.scale_levels)))
        }

        # Count equations by domain and scale
        for module, functions in equations.items():
            # Initialize module-specific counters
            coverage["physics_by_module"][module] = {domain: 0 for domain in self.physics_domains}
            coverage["scale_by_module"][module] = {level: 0 for level in self.scale_levels}

            for func_name, func_info in functions.items():
                physics_domain = func_info["physics_domain"]
                application_domain = func_info["application_domain"]
                scale_level = func_info["scale_level"]

                # Increment counters
                if physics_domain in coverage["physics_domains"]:
                    coverage["physics_domains"][physics_domain] += 1
                    coverage["physics_by_module"][module][physics_domain] += 1

                if application_domain in coverage["application_domains"]:
                    coverage["application_domains"][application_domain] += 1

                if scale_level in coverage["scale_levels"]:
                    coverage["scale_levels"][scale_level] += 1
                    coverage["scale_by_module"][module][scale_level] += 1

                # Update cross-domain coverage
                if physics_domain in self.physics_domains and application_domain in self.application_domains:
                    physics_idx = self.physics_domains.index(physics_domain)
                    app_idx = self.application_domains.index(application_domain)
                    coverage["cross_domain_coverage"][physics_idx, app_idx] += 1

                # Update cross-scale coverage
                if physics_domain in self.physics_domains and scale_level in self.scale_levels:
                    physics_idx = self.physics_domains.index(physics_domain)
                    scale_idx = self.scale_levels.index(scale_level)
                    coverage["cross_scale_coverage"][physics_idx, scale_idx] += 1

        # Calculate overall coverage metrics
        total_physics_domains = len(self.physics_domains)
        covered_physics_domains = sum(1 for count in coverage["physics_domains"].values() if count > 0)
        coverage["physics_coverage_pct"] = 100 * covered_physics_domains / total_physics_domains

        total_application_domains = len(self.application_domains)
        covered_application_domains = sum(1 for count in coverage["application_domains"].values() if count > 0)
        coverage["application_coverage_pct"] = 100 * covered_application_domains / total_application_domains

        total_scale_levels = len(self.scale_levels)
        covered_scale_levels = sum(1 for count in coverage["scale_levels"].values() if count > 0)
        coverage["scale_coverage_pct"] = 100 * covered_scale_levels / total_scale_levels

        return coverage

    def _identify_gaps(self, coverage):
        """
        Identify gaps in equation coverage.

        Args:
            coverage: Dictionary of coverage metrics

        Returns:
            List of gap descriptions
        """
        gaps = []

        # Check physics domain gaps
        for domain, count in coverage["physics_domains"].items():
            if count == 0:
                gaps.append({
                    "type": "physics_domain",
                    "domain": domain,
                    "description": f"No equations found for physics domain: {domain}",
                    "severity": "high",
                    "modules": [],
                    "recommendation": f"Add equations modeling {domain} principles"
                })
            elif count < 2:
                gaps.append({
                    "type": "physics_domain",
                    "domain": domain,
                    "description": f"Limited coverage of physics domain: {domain} (only {count} equation)",
                    "severity": "medium",
                    "modules": [],
                    "recommendation": f"Expand {domain} modeling with additional equations"
                })

        # Check expected equations
        for domain, expected in self.expected_equations.items():
            for equation in expected:
                found = False
                modules_with_equation = []

                for module, functions in coverage.get("physics_by_module", {}).items():
                    if functions.get(domain, 0) > 0:
                        # Module has equations in this domain, check if it has this specific equation
                        if equation in coverage.get("equations", {}).get(module, {}):
                            found = True
                            modules_with_equation.append(module)

                if not found:
                    gaps.append({
                        "type": "expected_equation",
                        "domain": domain,
                        "equation": equation,
                        "description": f"Missing expected equation: {equation} in {domain} domain",
                        "severity": "high",
                        "modules": modules_with_equation,
                        "recommendation": f"Implement {equation} function"
                    })

        # Check cross-domain gaps
        cross_domain = coverage.get("cross_domain_coverage", np.zeros((1, 1)))
        for i, physics in enumerate(self.physics_domains):
            for j, application in enumerate(self.application_domains):
                if i < cross_domain.shape[0] and j < cross_domain.shape[1] and cross_domain[i, j] == 0:
                    # No equations connecting this physics domain to this application domain
                    gaps.append({
                        "type": "cross_domain",
                        "physics": physics,
                        "application": application,
                        "description": f"No equations connecting {physics} physics to {application} application",
                        "severity": "medium",
                        "modules": [],
                        "recommendation": f"Create equation relating {physics} principles to {application} dynamics"
                    })

        # Check cross-scale gaps
        cross_scale = coverage.get("cross_scale_coverage", np.zeros((1, 1)))
        for i, physics in enumerate(self.physics_domains):
            for j, scale in enumerate(self.scale_levels):
                if i < cross_scale.shape[0] and j < cross_scale.shape[1] and cross_scale[i, j] == 0:
                    # No equations applying this physics domain to this scale
                    gaps.append({
                        "type": "cross_scale",
                        "physics": physics,
                        "scale": scale,
                        "description": f"No equations applying {physics} physics at {scale} scale",
                        "severity": "medium",
                        "modules": [],
                        "recommendation": f"Create equation applying {physics} principles at {scale} scale"
                    })

        # Check transitions between adjacent scales
        for i in range(len(self.scale_levels) - 1):
            scale1 = self.scale_levels[i]
            scale2 = self.scale_levels[i + 1]

            has_transition = False
            for module, functions in coverage.get("equations", {}).items():
                for func_name, func_info in functions.items():
                    # Look for functions that mention both scales
                    if (scale1 in func_info["docstring"].lower() and scale2 in func_info["docstring"].lower()) or \
                            (scale1 in func_name.lower() and scale2 in func_name.lower()):
                        has_transition = True
                        break

            if not has_transition:
                gaps.append({
                    "type": "scale_transition",
                    "scale1": scale1,
                    "scale2": scale2,
                    "description": f"Missing transition equations between {scale1} and {scale2} scales",
                    "severity": "high",
                    "modules": [],
                    "recommendation": f"Create transition equations connecting {scale1} and {scale2} scales"
                })

        return gaps

    def _analyze_integration(self, equations):
        """
        Analyze cross-domain integration of equations.

        Args:
            equations: Dictionary of equations

        Returns:
            Dictionary of integration metrics
        """
        integration = {
            "cross_domain_functions": [],
            "cross_scale_functions": [],
            "integration_quality": {}
        }

        # Find functions that span multiple domains or scales
        for module, functions in equations.items():
            for func_name, func_info in functions.items():
                physics_domain = func_info["physics_domain"]
                application_domain = func_info["application_domain"]
                scale_level = func_info["scale_level"]

                # Check if function integrates multiple domains
                if physics_domain != "unknown":
                    docstring = func_info["docstring"].lower()
                    source = func_info["source_length"]

                    # Count references to other physics domains
                    domain_references = {}
                    for domain in self.physics_domains:
                        if domain != physics_domain:
                            # Check for references to this domain in docstring
                            if domain.replace("_", " ") in docstring:
                                domain_references[domain] = docstring.count(domain.replace("_", " "))

                    if domain_references:
                        integration["cross_domain_functions"].append({
                            "function": func_name,
                            "primary_domain": physics_domain,
                            "referenced_domains": domain_references,
                            "module": module
                        })

                # Check if function integrates multiple scales
                if scale_level != "unknown":
                    docstring = func_info["docstring"].lower()

                    # Count references to other scale levels
                    scale_references = {}
                    for scale in self.scale_levels:
                        if scale != scale_level:
                            # Check for references to this scale in docstring
                            if scale.replace("_", " ") in docstring:
                                scale_references[scale] = docstring.count(scale.replace("_", " "))

                    if scale_references:
                        integration["cross_scale_functions"].append({
                            "function": func_name,
                            "primary_scale": scale_level,
                            "referenced_scales": scale_references,
                            "module": module
                        })

        # Assess integration quality
        # More cross-domain and cross-scale functions indicate better integration
        num_cross_domain = len(integration["cross_domain_functions"])
        num_cross_scale = len(integration["cross_scale_functions"])
        total_functions = sum(len(funcs) for funcs in equations.values())

        if total_functions > 0:
            integration["cross_domain_ratio"] = num_cross_domain / total_functions
            integration["cross_scale_ratio"] = num_cross_scale / total_functions

            # Heuristic for integration quality
            integration["integration_quality"]["overall"] = 0.5 * integration["cross_domain_ratio"] + 0.5 * integration[
                "cross_scale_ratio"]

            # Integration quality by domain
            for domain in self.physics_domains:
                domain_funcs = sum(1 for func in integration["cross_domain_functions"]
                                   if func["primary_domain"] == domain)
                # Avoid division by zero
                total_domain_funcs = sum(1 for module, funcs in equations.items()
                                         for func_name, func_info in funcs.items()
                                         if func_info["physics_domain"] == domain)

                if total_domain_funcs > 0:
                    integration["integration_quality"][domain] = domain_funcs / total_domain_funcs
                else:
                    integration["integration_quality"][domain] = 0

            # Integration quality by scale
            for scale in self.scale_levels:
                scale_funcs = sum(1 for func in integration["cross_scale_functions"]
                                  if func["primary_scale"] == scale)
                # Avoid division by zero
                total_scale_funcs = sum(1 for module, funcs in equations.items()
                                        for func_name, func_info in funcs.items()
                                        if func_info["scale_level"] == scale)

                if total_scale_funcs > 0:
                    integration["integration_quality"][scale] = scale_funcs / total_scale_funcs
                else:
                    integration["integration_quality"][scale] = 0

        return integration

    def visualize_coverage(self, coverage, output_file=None):
        """
        Visualize equation coverage across domains and scales.

        Args:
            coverage: Dictionary of coverage metrics
            output_file: Path to save the visualization

        Returns:
            matplotlib figure
        """
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 14))

        # Plot physics domain coverage
        ax = axes[0, 0]
        domains = list(coverage["physics_domains"].keys())
        counts = list(coverage["physics_domains"].values())
        ax.bar(domains, counts)
        ax.set_title("Equation Coverage by Physics Domain")
        ax.set_xlabel("Physics Domain")
        ax.set_ylabel("Number of Equations")
        ax.tick_params(axis='x', rotation=45)

        # Plot application domain coverage
        ax = axes[0, 1]
        domains = list(coverage["application_domains"].keys())
        counts = list(coverage["application_domains"].values())
        ax.bar(domains, counts)
        ax.set_title("Equation Coverage by Application Domain")
        ax.set_xlabel("Application Domain")
        ax.set_ylabel("Number of Equations")
        ax.tick_params(axis='x', rotation=45)

        # Plot scale level coverage
        ax = axes[1, 0]
        scales = list(coverage["scale_levels"].keys())
        counts = list(coverage["scale_levels"].values())
        ax.bar(scales, counts)
        ax.set_title("Equation Coverage by Scale Level")
        ax.set_xlabel("Scale Level")
        ax.set_ylabel("Number of Equations")
        ax.tick_params(axis='x', rotation=45)

        # Plot cross-domain coverage heatmap
        ax = axes[1, 1]
        cmap = plt.cm.Blues
        im = ax.imshow(coverage["cross_domain_coverage"], cmap=cmap)
        ax.set_title("Cross-Domain Coverage")
        ax.set_xlabel("Application Domain")
        ax.set_ylabel("Physics Domain")
        ax.set_xticks(np.arange(len(self.application_domains)))
        ax.set_yticks(np.arange(len(self.physics_domains)))
        ax.set_xticklabels(self.application_domains)
        ax.set_yticklabels(self.physics_domains)
        ax.tick_params(axis='x', rotation=45)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Number of Equations")

        # Adjust layout
        plt.tight_layout()

        # Save figure if output file provided
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Coverage visualization saved to {output_file}")

        return fig

    def visualize_gaps(self, gaps, output_file=None):
        """
        Visualize identified gaps in equation coverage.

        Args:
            gaps: List of gap descriptions
            output_file: Path to save the visualization

        Returns:
            matplotlib figure
        """
        if not gaps:
            logger.warning("No gaps to visualize")
            return None

        # Count gaps by type and severity
        gap_types = {}
        gap_severity = {"high": 0, "medium": 0, "low": 0}

        for gap in gaps:
            gap_type = gap["type"]
            severity = gap.get("severity", "medium")

            gap_types[gap_type] = gap_types.get(gap_type, 0) + 1
            gap_severity[severity] = gap_severity.get(severity, 0) + 1

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot gaps by type
        types = list(gap_types.keys())
        counts = list(gap_types.values())
        ax1.bar(types, counts)
        ax1.set_title("Equation Gaps by Type")
        ax1.set_xlabel("Gap Type")
        ax1.set_ylabel("Number of Gaps")
        ax1.tick_params(axis='x', rotation=45)

        # Plot gaps by severity
        severities = list(gap_severity.keys())
        counts = list(gap_severity.values())
        colors = ['red', 'orange', 'yellow']
        ax2.bar(severities, counts, color=colors)
        ax2.set_title("Equation Gaps by Severity")
        ax2.set_xlabel("Severity")
        ax2.set_ylabel("Number of Gaps")

        # Adjust layout
        plt.tight_layout()

        # Save figure if output file provided
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Gap visualization saved to {output_file}")

        return fig


# Example usage
if __name__ == "__main__":
    analyzer = EquationCoverageAnalyzer()

    # Analyze equation coverage
    results = analyzer.analyze_equation_set()

    # Visualize coverage
    analyzer.visualize_coverage(results["coverage"], "validation/reports/unified/equation_coverage.png")

    # Visualize gaps
    if results["gaps"]:
        analyzer.visualize_gaps(results["gaps"], "validation/reports/unified/equation_gaps.png")

    # Print summary
    print(f"Physics domain coverage: {results['coverage']['physics_coverage_pct']:.1f}%")
    print(f"Application domain coverage: {results['coverage']['application_coverage_pct']:.1f}%")
    print(f"Scale level coverage: {results['coverage']['scale_coverage_pct']:.1f}%")
    print(f"Identified {len(results['gaps'])} gaps in equation coverage")

    if __name__ == "__main__":
        analyzer = EquationCoverageAnalyzer()

        # Specifically analyze strong nuclear
        strong_nuclear_results = analyzer.analyze_equation_set(
            equation_modules=["physics_domains.strong_nuclear"],
            identify_gaps=False
        )

        # Print detected functions
        print("\nStrong Nuclear Functions Detected:")
        for module, functions in strong_nuclear_results["equations"].items():
            for func_name, func_info in functions.items():
                print(f"  - {func_name} ({func_info['physics_domain']})")