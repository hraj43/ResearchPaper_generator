import re
from typing import Any, Dict, List
from llama_index.llms.openai import OpenAI
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, Context, step
from llama_index.core.workflow import Event

def extract_title(outline):
    """Function to extract the title from the first line of the outline"""
    first_line = outline.strip().split('\n')[0]
    return first_line.strip('# ').strip()

def generate_query_with_llm(llm, title, section, subsection):
    """Function to generate a query for a report using LLM"""
    prompt = f"Generate a research query for a report on {title}. "
    prompt += f"The query should be for the subsection '{subsection}' under the main section '{section}'. "
    prompt += "The query should guide the research to gather relevant information for this part of the report. The query should be clear, short and concise. "

    response = llm.complete(prompt)
    return str(response).strip()

def classify_query(llm, query):
    """Function to classify the query as either 'LLM' or 'INDEX' based on the query content"""
    prompt = f"""Classify the following query as either "LLM" if it can be answered directly by a large language model with general knowledge, or "INDEX" if it likely requires querying an external index or database for specific or up-to-date information.

    Query: "{query}"

    Consider the following:
    1. If the query asks for general knowledge, concepts, or explanations, classify as "LLM".
    2. If the query asks for specific facts, recent events, or detailed information that might not be in the LLM's training data, classify as "INDEX".
    3. If unsure, err on the side of "INDEX".

    Classification:"""

    classification = str(llm.complete(prompt)).strip().upper()
    if classification not in ["LLM", "INDEX"]:
        classification = "INDEX"  # Default to INDEX if the response is unclear
    return classification

def parse_outline_and_generate_queries(llm, outline):
    """Function to parse the outline and generate queries for each section and subsection"""
    lines = outline.strip().split('\n')
    title = extract_title(outline)
    current_section = ""
    queries = {}

    for line in lines[1:]:  # Skip the title line
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        
        if line.startswith('## '):
            current_section = line.strip('# ').strip()
            queries[current_section] = {}
        elif re.match(r'^\d+\.\d+\.', line) or re.match(r'^\d+\.\d+\s', line):
            subsection = line.strip()
            query = generate_query_with_llm(llm, title, current_section, subsection)
            classification = classify_query(llm, query)
            queries[current_section][subsection] = {"query": query, "classification": classification}

    # Handle sections without subsections
    for section in queries:
        if not queries[section]:
            query = generate_query_with_llm(llm, title, section, "General overview")
            classification = classify_query(llm, query)
            queries[section]["General"] = {"query": query, "classification": classification}

    return queries, title


class ReportGenerationEvent(Event):
    pass


class ReportGenerationAgent(Workflow):
    """Report generation agent."""

    def __init__(
        self,
        query_engine: Any,
        llm: FunctionCallingLLM | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.query_engine = query_engine
        self.llm = llm or OpenAI(model='gpt-3.5-turbo')
        self.debug = kwargs.get('verbose', False)

    def log(self, message):
        """Print debug messages if verbose mode is enabled"""
        if self.debug:
            print(f"[ReportAgent] {message}")

    def format_report(self, section_contents, title, outline):
        """Format the report based on the section contents."""
        # Extract structure from outline
        sections = self._parse_outline_structure(outline)
        
        self.log(f"Formatting report with title: {title}")
        self.log(f"Found {len(sections)} sections in outline")
        
        # Start with the title
        report = f"# {title}\n\n"
        
        # Process each section in order according to the outline
        for section_info in sections:
            section_num = section_info['number']
            section_title = section_info['title']
            section_key = f"{section_num} {section_title}"
            
            self.log(f"Processing section: {section_key}")
            
            # Check if this is an introduction or conclusion section
            is_intro = "introduction" in section_title.lower()
            is_conclusion = "conclusion" in section_title.lower()
            
            if is_intro:
                self.log("Generating introduction")
                # Generate introduction based on overall report content
                intro_content = self._generate_introduction(title, section_contents)
                report += f"## {section_num} {section_title}\n\n{intro_content}\n\n"
            elif is_conclusion:
                self.log("Generating conclusion")
                # Generate conclusion based on overall report content
                conclusion_content = self._generate_conclusion(title, section_contents)
                report += f"## {section_num} {section_title}\n\n{conclusion_content}\n\n"
            else:
                # Regular section
                if section_key in section_contents:
                    # Add section heading
                    report += f"## {section_num} {section_title}\n\n"
                    
                    # Generate section overview if needed
                    if len(section_info.get('subsections', [])) > 0:
                        overview = self._generate_section_overview(section_key, section_contents[section_key])
                        report += f"{overview}\n\n"
                    
                    # Process subsections in order
                    for subsection in section_info.get('subsections', []):
                        subsection_num = subsection['number']
                        subsection_title = subsection['title']
                        subsection_key = f"{subsection_num} {subsection_title}"
                        
                        self.log(f"  Processing subsection: {subsection_key}")
                        
                        # Add content if available
                        if subsection_key in section_contents[section_key]:
                            content = section_contents[section_key][subsection_key]
                            report += f"### {subsection_num} {subsection_title}\n\n{content}\n\n"
                        else:
                            self.log(f"  Warning: No content for subsection {subsection_key}")
                            # Generate placeholder content
                            placeholder = self._generate_placeholder_content(title, section_title, subsection_title)
                            report += f"### {subsection_num} {subsection_title}\n\n{placeholder}\n\n"
                else:
                    self.log(f"Warning: No content for section {section_key}")
                    # Generate placeholder
                    placeholder = self._generate_placeholder_content(title, section_title, "")
                    report += f"## {section_num} {section_title}\n\n{placeholder}\n\n"
        
        return report

    def _parse_outline_structure(self, outline):
        """Parse the outline to extract section and subsection structure"""
        lines = outline.strip().split('\n')
        sections = []
        current_section = None
        
        for line in lines[1:]:  # Skip title
            line = line.strip()
            if not line:
                continue
                
            # Section line (e.g., "## 2. Blockchain Fundamentals")
            if line.startswith('## '):
                section_text = line.strip('# ').strip()
                section_match = re.match(r'^(\d+\.)\s*(.*)$', section_text)
                if section_match:
                    section_num, section_title = section_match.groups()
                    current_section = {
                        'number': section_num,
                        'title': section_title,
                        'subsections': []
                    }
                    sections.append(current_section)
                else:
                    # Handle section without number
                    current_section = {
                        'number': str(len(sections) + 1) + ".",
                        'title': section_text,
                        'subsections': []
                    }
                    sections.append(current_section)
            
            # Subsection line (e.g., "2.1. Basics of Blockchain Technology")
            elif current_section and (re.match(r'^\d+\.\d+\.', line) or re.match(r'^\d+\.\d+\s', line)):
                subsection_match = re.match(r'^(\d+\.\d+)\.?\s*(.*)$', line)
                if subsection_match:
                    subsection_num, subsection_title = subsection_match.groups()
                    current_section['subsections'].append({
                        'number': subsection_num + ".",
                        'title': subsection_title
                    })
        
        return sections

    def _generate_introduction(self, title, section_contents):
        """Generate introduction for the report"""
        # Extract content snippets from each section to inform the introduction
        content_snippets = []
        for section, subsections in section_contents.items():
            if not "introduction" in section.lower():
                # Get a sample from each section
                samples = list(subsections.values())[:2]  # Limit to first 2 subsections
                content_snippets.extend(samples)
        
        # Limit the amount of context to avoid token limits
        context = "\n".join(content_snippets[:5])  # Limit to 5 snippets
        
        prompt = f"""Write a thorough introduction for a research paper titled "{title}".
        
        The paper covers the following topics:
        {", ".join([s.split(" ", 1)[1] for s in section_contents.keys() if not "introduction" in s.lower()])}
        
        Based on these topics and the following content samples:
        {context}
        
        Write an engaging introduction that:
        1. Introduces the topic and its importance
        2. Outlines the scope of the paper
        3. Provides context for the reader
        4. Sets up the structure of the following sections
        
        Keep the introduction comprehensive yet concise."""
        
        introduction = str(self.llm.complete(prompt))
        return introduction

    def _generate_conclusion(self, title, section_contents):
        """Generate conclusion for the report"""
        # Extract content snippets from each section to inform the conclusion
        content_snippets = []
        for section, subsections in section_contents.items():
            if not "conclusion" in section.lower():
                # Get a sample from each section
                samples = list(subsections.values())[:1]  # Limit to first subsection
                content_snippets.extend(samples)
        
        # Limit the amount of context to avoid token limits
        context = "\n".join(content_snippets[:5])  # Limit to 5 snippets
        
        prompt = f"""Write a thorough conclusion for a research paper titled "{title}".
        
        The paper covered the following topics:
        {", ".join([s.split(" ", 1)[1] for s in section_contents.keys() if not "conclusion" in s.lower() and not "introduction" in s.lower()])}
        
        Based on these topics and the following content samples:
        {context}
        
        Write a comprehensive conclusion that:
        1. Summarizes the key findings and insights
        2. Discusses the implications of the research
        3. Suggests potential areas for future research
        4. Ends with a meaningful closing statement
        
        The conclusion should be thorough and thoughtful."""
        
        conclusion = str(self.llm.complete(prompt))
        return conclusion

    def _generate_section_overview(self, section, subsection_contents):
        """Generate an overview for a section based on its subsections"""
        # Compile brief snippets from each subsection
        subsection_samples = "\n".join([
            f"{sub_key}: {sub_content[:200]}..." 
            for sub_key, sub_content in list(subsection_contents.items())[:3]
        ])
        
        section_title = section.split(" ", 1)[1] if " " in section else section
        
        prompt = f"""Write a brief overview paragraph for the section "{section_title}" of a research paper.
        
        The section contains the following subsections:
        {", ".join([sub_key.split(" ", 1)[1] if " " in sub_key else sub_key for sub_key in subsection_contents.keys()])}
        
        Based on these subsection samples:
        {subsection_samples}
        
        Write a concise paragraph that introduces this section and ties together the subsections that follow.
        The paragraph should be no more than 3-5 sentences."""
        
        overview = str(self.llm.complete(prompt))
        return overview

    def _generate_placeholder_content(self, report_title, section_title, subsection_title):
        """Generate placeholder content for missing sections/subsections"""
        if subsection_title:
            prompt = f"""Generate content for the subsection "{subsection_title}" under the section "{section_title}" for a research paper titled "{report_title}".
            
            The content should be informative, well-structured, and around 200-300 words. Focus on providing accurate and useful information related to the subsection topic."""
        else:
            prompt = f"""Generate content for the section "{section_title}" for a research paper titled "{report_title}".
            
            The content should be informative, well-structured, and around 300-400 words. Provide a comprehensive overview of the topic covered by this section."""
        
        content = str(self.llm.complete(prompt))
        return content

    def generate_section_content(self, queries):
        """Generate content for each section and subsection in the outline."""
        self.log("Generating content for sections and subsections")
        section_contents = {}
        
        for section, subsections in queries.items():
            self.log(f"Generating content for section: {section}")
            section_contents[section] = {}
            
            for subsection, data in subsections.items():
                self.log(f"  Processing subsection: {subsection}")
                query = data['query']
                classification = data['classification']
                
                self.log(f"  Query classification: {classification}")
                self.log(f"  Query: {query}")
                
                try:
                    if classification == "LLM":
                        expanded_query = f"""Based on the query: "{query}"
                        
                        Provide a comprehensive response that would be suitable for a subsection of a research paper. 
                        The response should be well-structured, informative, and around 300-500 words.
                        Include relevant facts, concepts, and examples where appropriate.
                        Ensure the content is cohesive and flows well as part of a larger document."""
                        
                        answer = str(self.llm.complete(expanded_query))
                    else:
                        # Add instructions to format the response appropriately
                        query_with_instructions = f"""Query: {query}
                        
                        Please provide a comprehensive response suitable for a research paper subsection.
                        Include specific details, facts, and references where possible."""
                        
                        answer = str(self.query_engine.query(query_with_instructions))
                    
                    # Handle potentially empty responses
                    if not answer or len(answer.strip()) < 50:
                        self.log(f"  Warning: Short or empty response received")
                        # Generate fallback content with LLM
                        fallback_query = f"Provide informative content about {subsection} for a research paper on {section}"
                        answer = str(self.llm.complete(fallback_query))
                    
                    section_contents[section][subsection] = answer
                    self.log(f"  Content generated: {len(answer)} characters")
                    
                except Exception as e:
                    self.log(f"  Error generating content: {str(e)}")
                    # Generate fallback content
                    section_contents[section][subsection] = f"Content could not be generated for this subsection due to an error: {str(e)}"
        
        return section_contents

    @step(pass_context=True)
    async def queries_generation_event(self, ctx: Context, ev: StartEvent) -> ReportGenerationEvent:
        """Generate queries for the report."""
        self.log("Starting queries generation event")
        ctx.data["outline"] = ev.outline
        queries, title = parse_outline_and_generate_queries(self.llm, ctx.data["outline"])
        ctx.data["title"] = title
        
        self.log(f"Generated queries for {len(queries)} sections")
        return ReportGenerationEvent(queries=queries)

    @step(pass_context=True)
    async def generate_report(
        self, ctx: Context, ev: ReportGenerationEvent
    ) -> StopEvent:
        """Generate report."""
        self.log("Starting report generation")
        queries = ev.queries
        title = ctx.data.get("title", "Research Paper")
        
        # Generate contents for all sections
        section_contents = self.generate_section_content(queries)
        
        # Format and compile the final report
        report = self.format_report(section_contents, title, ctx.data["outline"])
        
        self.log("Report generation completed")
        return StopEvent(result={"response": report})