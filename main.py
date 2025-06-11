import logging
from dataclasses import dataclass
from typing_extensions import TypedDict
from typing import Dict, Any, Optional
from google import genai
from google.genai import types
import os, json
from langgraph.graph import StateGraph, END
from fastapi import FastAPI, HTTPException
import uvicorn
from datetime import datetime
from langchain_community.utilities.sql_database import SQLDatabase
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from pydantic import BaseModel
from google.api_core.exceptions import TooManyRequests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for name in logging.root.manager.loggerDict:
    if name != "__main__":
        logging.getLogger(name).setLevel(logging.WARNING)

# Database configuration
with open('config.json', 'r') as f:
    DB_CONFIG = json.load(f)

# State definition for LangGraph
class AgentState(TypedDict):
    customer_id: str
    customer_details: Optional[str]
    purchase_patterns: Optional[str]
    product_affinities: Optional[str]
    opportunity_scores: Optional[str]
    research_report: Optional[Any]
    error: Optional[str]

class SQLAgent:
    def __init__(self, dbconfig: Dict, model: str = 'gemini-2.0-flash-lite'):
        """Initialize the SQLAgent with database configuration and model."""
        
        pg_uri = f"postgresql+psycopg2://{dbconfig['user']}:{dbconfig['password']}@{dbconfig['host']}:{dbconfig['port']}/{dbconfig['database']}"
        db = SQLDatabase.from_uri(pg_uri)
        
        if not os.getenv("Gemini_API_Key"):
            from getpass import getpass
            os.environ["Gemini_API_Key"] = getpass("Enter API key for Google Gemini: ")
        
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.getenv("Gemini_API_Key"),
        )
        
        prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
        system_message = prompt_template.format(dialect="PostgreSQL", top_k=3)
        
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        
        self.agent_executor = create_react_agent(llm, toolkit.get_tools(), prompt=system_message)
    
    def query(self, query: str) -> dict[str, Any] | Any:
        """Execute a query to the database via the SQLAgent and return the result."""
        
        try:
            return self.agent_executor.invoke(
                {"messages": [("system", "Do as the user says. Replace any question marks in the data where there's a monetary value with the $ symbol. Provide all the results in a structured format, and don't include the SQL query in the output. Don't limit the SQL output anywhere."), ("user", query)]}
            )
        except Exception as e:
            logger.error(f"SQL query error: {e}")
            raise

class CustomerContextAgent:
    def __init__(self, sql_agent: SQLAgent):
        """Initialize the CustomerContextAgent with an SQLAgent instance."""
        self.sql_agent = sql_agent
    
    def extract_customer_details(self, state: AgentState) -> AgentState:
        """Extract customer details and purchase history from the database."""
        try:
            customer_id = state["customer_id"]
            logger.info(f"Extracting details for customer: {customer_id}")
            
            response = self.sql_agent.query(f"Extract the entire customer profile for {customer_id} from table 'customer_data'. Include columns: customer_id, customer_name, industry, annual_revenue_usd, number_of_employees, customer_priority, customer_rating, account_type, location, current_products, product_usage_percent, cross_sell_synergy, last_activity_date, opportunity_stage. Also extract the entire customer purchase history for {customer_id} from table 'customer_data'. Include columns: product, quantity, unit_price_usd, total_price_usd, purchase_date.")
            
            messages = response['messages']
            ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
            detail = ai_messages[-1].content if ai_messages else None
            
            state["customer_details"] = detail
            logger.info(f"Successfully extracted details for {customer_id}")
            
        except Exception as e:
            logger.error(f"Error in CustomerContextAgent: {e}")
            state["error"] = str(e)
        
        return state

class PurchasePatternAnalysis:
    def __init__(self, sql_agent: SQLAgent):
        """Initialize the PurchasePatternAnalysis with an SQLAgent instance."""
        self.sql_agent = sql_agent
    
    def analyze_patterns(self, state: AgentState) -> AgentState:
        """Analyze purchase patterns for the customer and identify frequent products, spending, and opportunities."""
        try:
            if state.get("error") or not state.get("customer_details"):
                return state
            
            logger.info(f"Analyzing purchase patterns for customer {state['customer_id']}")
            
            customer_details = state["customer_details"]
            
            response = self.sql_agent.query(f"Here's the details for a customer: {customer_details}. Analyze the purchase patterns of this customer. Identify frequent products, product spending, missing opportunities, and calculate purchase frequency score and average order value.")
            
            messages = response['messages']
            ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
            patterns = ai_messages[-1].content if ai_messages else None
            
            state["purchase_patterns"] = patterns
            logger.info(f"Analyzed purchase patterns for customer {state['customer_id']}")
            
        except Exception as e:
            logger.error(f"Error in PurchasePatternAnalysis: {e}")
            state["error"] = str(e)
        
        return state

class ProductAffinityAgent:
    def __init__(self, sql_agent: SQLAgent):
        """Initialize the ProductAffinityAgent with an SQLAgent instance."""
        self.sql_agent = sql_agent
    
    def suggest_related_products(self, state: AgentState) -> AgentState:
        """Suggest related products based on customer purchase patterns and affinities."""
        try:
            if state.get("error") or not state.get("customer_details") or not state.get("purchase_patterns"):
                return state
            
            logger.info(f"Generating related product suggestions for customer {state['customer_id']}")
            
            customer_details = state["customer_details"]
            patterns = state["purchase_patterns"]
            
            response = self.sql_agent.query(f"""Here's the details for a customer: {customer_details}.
Here's the purchase patterns for the same customer: {patterns}.
Analyze the product affinities for this customer. Identify cross-sell and upsell opportunities based on their purchase history and patterns. Provide a structured output with cross-sell and upsell opportunities, including product names and potential revenue impact.""")
            
            messages = response['messages']
            ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
            affinities = ai_messages[-1].content if ai_messages else None
            
            state["product_affinities"] = affinities
            logger.info(f"Generated related product suggestions for customer {state['customer_id']}")
            
        except Exception as e:
            logger.error(f"Error in ProductAffinityAgent: {e}")
            state["error"] = str(e)
        
        return state

class OpportunityScoringAgent:
    def __init__(self, sql_agent: SQLAgent):
        """Initialize the OpportunityScoringAgent with an SQLAgent instance."""
        self.sql_agent = sql_agent
    
    def score_opportunities(self, state: AgentState) -> AgentState:
        """Score cross-sell and upsell opportunities based on purchase patterns and product affinities."""
        try:
            if state.get("error") or not state.get("customer_details") or not state.get("purchase_patterns") or not state.get("product_affinities"):
                return state

            logger.info(f"Scoring cross-sell and upsell opportunities for customer {state['customer_id']}")
            
            customer_details = state["customer_details"]
            patterns = state["purchase_patterns"]
            affinities = state["product_affinities"]
            
            response = self.sql_agent.query(f"""Here's the details for a customer: {customer_details}.
Here's the purchase patterns for the same customer: {patterns}.
Here's the product affinities for the same customer: {affinities}.
Analyze the cross-sell and upsell opportunities for this customer. Score the opportunities based on purchase frequency, product affinities, and potential revenue impact. Provide a structured output with cross-sell and upsell scores, including reasoning for each score.""")
            
            messages = response['messages']
            ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
            scores = ai_messages[-1].content if ai_messages else None
            
            state["opportunity_scores"] = scores
            logger.info(f"Scored cross-sell and upsell opportunities for customer {state['customer_id']}")
            
        except Exception as e:
            logger.error(f"Error in OpportunityScoringAgent: {e}")
            state["error"] = str(e)
        
        return state

class Output(BaseModel):
    customer_id: str
    research_report: str
    recommendation_list: list[str]

class RecommendationReportAgent:
    def generate_report(self, state: AgentState) -> AgentState:
        """Generate a comprehensive recommendation report based on customer data and analysis results."""
        
        logger.info("Generating recommendation report...")
        
        format = """Research Report: Cross-Sell and Upsell Opportunities for Acme Corp
Introduction:
This report analyzes recent purchasing behavior of Acme Corp and benchmarks against
industry peers to identify cross-sell and upsell opportunities.
Customer Overview:
- Industry: Construction
- Annual Revenue: $1.5M
- Recent Purchases: Generators, Drills
Data Analysis:
- Purchase patterns indicate frequent purchases of generators, but limited spending on
accessory products.
- Industry peers commonly purchase backup batteries and safety gear.
- Product affinity analysis suggests drill bits and protective gloves as complementary
products.
Recommendations:
1. Offer Backup Batteries as a cross-sell opportunity for Generators.
2. Promote Drill Bits and Protective Gloves as upsell items to existing Drill owners.
3. Explore Heavy Equipment Rental services popular in the construction sector.
Conclusion:
Targeted cross-sell and upsell campaigns focusing on these products can increase revenue
and customer satisfaction."""
        
        client = genai.Client(api_key = os.getenv("Gemini_API_Key"))
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                config=types.GenerateContentConfig(
                    response_mime_type= "application/json",
                    response_schema= Output,
                    system_instruction="You are an expert business and data analyst. Generate a comprehensive report and recommendation list following the given sample report based on the provided data (customer data and analysis results) unless the data seems empty or faulty. In that case, just say 'Error: No data available for analysis.'.",
                ),
                contents=f"""REPORT SAMPLE: {format}
DATA: {state}"""
            )
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            state["error"] = str(e)
            return state
        
        logger.info("Recommendation report generated successfully")
        
        self._create_report_file(response.text)
        
        state["research_report"] = response.text
        return state
    
    def _create_report_file(self, report):
        with open("recommendation_report.txt", "w") as file:
            file.write(report)
        logger.info("Recommendation report saved to recommendation_report.txt")

class CrossSellUpsellAgent:
    def __init__(self, db_config: Dict):
        self.sql_agent = SQLAgent(db_config)
        self.customer_agent = CustomerContextAgent(self.sql_agent)
        self.pattern_agent = PurchasePatternAnalysis(self.sql_agent)
        self.affinity_agent = ProductAffinityAgent(self.sql_agent)
        self.scoring_agent = OpportunityScoringAgent(self.sql_agent)
        self.report_agent = RecommendationReportAgent()
        
        # Build the graph
        self.workflow = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("customer_context_node", self.customer_agent.extract_customer_details)
        workflow.add_node("purchase_patterns_node", self.pattern_agent.analyze_patterns)
        workflow.add_node("product_affinity_node", self.affinity_agent.suggest_related_products)
        workflow.add_node("opportunity_scoring_node", self.scoring_agent.score_opportunities)
        workflow.add_node("generate_report_node", self.report_agent.generate_report)
        
        # Define the flow
        workflow.set_entry_point("customer_context_node")
        workflow.add_edge("customer_context_node", "purchase_patterns_node")
        workflow.add_edge("purchase_patterns_node", "product_affinity_node")
        workflow.add_edge("product_affinity_node", "opportunity_scoring_node")
        workflow.add_edge("opportunity_scoring_node", "generate_report_node")
        workflow.add_edge("generate_report_node", END)
        
        return workflow.compile()
    
    def run_analysis(self, customer_id: str) -> Dict[str, Any]:
        initial_state = AgentState(customer_id=customer_id)
        result = self.workflow.invoke(initial_state)
        
        if result.get("error"):
            raise Exception(result["error"])
        
        return result.get("research_report")

app = FastAPI(title="Cross-Sell/Upsell Recommendation API")
agent = CrossSellUpsellAgent(DB_CONFIG)

@app.get("/recommendation")
async def get_recommendations(customer_id: str):
    try:
        result = agent.run_analysis(customer_id)
        return result
    except TooManyRequests as e:
        logger.error(f"Rate limit exceeded for gemini api: {e}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    except Exception as e:
        logger.error(f"API Error for customer {customer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)