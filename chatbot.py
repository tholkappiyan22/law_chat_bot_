from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import logging
import time
from datetime import datetime
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Gemini AI
GEMINI_API_KEY = "AIzaSyCFuIgj2_EcQN1qy3omOjkGcwGnTYFOMYg"
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the model - use the latest available model
model = None

# Global variable to track service readiness
service_ready = False

class LegalChatbot:
    def __init__(self):
        self.conversation_history = []
        self.max_history = 10  # Keep last 10 exchanges for context
        
    def create_legal_prompt(self, user_question):
        """Create a specialized prompt for legal queries"""
        system_prompt = """You are an AI legal assistant designed to provide helpful legal information and guidance. 

IMPORTANT GUIDELINES:
- Provide accurate, helpful legal information in simple language
- Always include disclaimers that this is not legal advice and users should consult qualified lawyers
- Focus on Indian law and legal system unless specifically asked about other jurisdictions
- Be empathetic and understanding, especially for sensitive legal matters
- Provide practical next steps when possible
- If you don't know something, admit it and suggest consulting a legal professional
- Keep responses concise but comprehensive
- Use bullet points or numbered lists for clarity when appropriate

LEGAL AREAS TO COVER:
- Consumer rights and protection
- Women's rights and domestic violence
- Child protection and family law
- Property and real estate law
- Employment and labor law
- Criminal law basics
- Civil rights and procedures
- Digital/cyber law
- Contract law basics
- Government schemes and legal aid

Remember: You are providing legal information, not legal advice. Always recommend consulting with a qualified lawyer for specific legal situations."""

        # Add conversation history for context
        context = ""
        if self.conversation_history:
            context = "\n\nRecent conversation context:\n"
            for i, (q, a) in enumerate(self.conversation_history[-3:], 1):  # Last 3 exchanges
                context += f"{i}. Q: {q[:100]}...\n   A: {a[:100]}...\n"

        full_prompt = f"{system_prompt}{context}\n\nUser Question: {user_question}\n\nResponse:"
        return full_prompt

    def get_response(self, user_question):
        """Get response from Gemini AI with legal context"""
        global model
        try:
            if not model:
                return "AI model is not initialized. Please try again later."
                
            # Create legal-specific prompt
            prompt = self.create_legal_prompt(user_question)
            
            # Generate response
            response = model.generate_content(prompt)
            
            if response and response.text:
                ai_response = response.text.strip()
                
                # Add legal disclaimer if not already present
                if not any(disclaimer in ai_response.lower() for disclaimer in 
                          ['disclaimer', 'not legal advice', 'consult a lawyer', 'legal professional']):
                    ai_response += "\n\n⚖️ **Legal Disclaimer:** This is general legal information only and not legal advice. Please consult with a qualified lawyer for advice specific to your situation."
                
                # Store in conversation history
                self.add_to_history(user_question, ai_response)
                
                return ai_response
            else:
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question or consult with a legal professional for assistance."
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'm experiencing technical difficulties. Please try again later or consult with a legal professional for immediate assistance."
    
    def add_to_history(self, question, answer):
        """Add exchange to conversation history"""
        self.conversation_history.append((question, answer))
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def summarize_text(self, text):
        """Simplify/summarize legal text for better understanding"""
        global model
        try:
            if not model:
                return "AI model is not initialized. Please try again later."
                
            summarize_prompt = f"""Please simplify and summarize the following legal text in simple, easy-to-understand language. 
Make it accessible to someone without legal background while retaining the key information:

Text to simplify: {text}

Simplified version:"""
            
            response = model.generate_content(summarize_prompt)
            if response and response.text:
                return response.text.strip()
            else:
                return "Unable to simplify text at this time."
                
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return "Unable to simplify text at this time."

# Initialize chatbot
legal_chatbot = LegalChatbot()

def initialize_service():
    """Initialize the service and test AI connection"""
    global service_ready, model
    try:
        logger.info("Initializing Legal AI Chatbot service...")
        
        # List of models to try in order of preference
        models_to_try = [
            'gemini-1.5-flash',
            'gemini-1.5-flash-latest', 
            'gemini-1.5-pro',
            'gemini-1.5-pro-latest',
            'gemini-2.0-flash',
            'gemini-2.5-flash'
        ]
        
        for model_name in models_to_try:
            try:
                logger.info(f"🔄 Trying model: {model_name}")
                test_model = genai.GenerativeModel(model_name)
                
                # Test the model with a simple prompt
                test_response = test_model.generate_content("Hello, respond with 'Working' if you can generate text.")
                
                if test_response and test_response.text:
                    model = test_model
                    logger.info(f"✅ Successfully initialized model: {model_name}")
                    logger.info("✅ Gemini AI connection successful")
                    service_ready = True
                    logger.info("🚀 Legal AI Chatbot service ready!")
                    return True
                    
            except Exception as model_error:
                logger.warning(f"❌ Failed to initialize {model_name}: {str(model_error)}")
                continue
        
        # If we get here, no model worked
        logger.error("❌ Failed to initialize any model")
        return False
            
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global service_ready
    
    if not service_ready:
        # Try to initialize if not ready
        initialize_service()
    
    if service_ready:
        return jsonify({
            'status': 'ready',
            'message': 'Legal AI Chatbot service is running',
            'timestamp': datetime.now().isoformat(),
            'model': 'gemini-pro'
        }), 200
    else:
        return jsonify({
            'status': 'initializing',
            'message': 'Service is starting up, please wait...',
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/ask', methods=['POST'])
def ask_legal_question():
    """Main endpoint for legal questions"""
    global service_ready
    
    if not service_ready:
        return jsonify({
            'error': 'Service is initializing. Please try again in a moment.'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'error': 'Please provide a question in the request body'
            }), 400
        
        user_question = data['question'].strip()
        
        if not user_question:
            return jsonify({
                'error': 'Question cannot be empty'
            }), 400
        
        # Log the question (for debugging, remove in production)
        logger.info(f"Received question: {user_question[:100]}...")
        
        # Get response from legal chatbot
        start_time = time.time()
        answer = legal_chatbot.get_response(user_question)
        response_time = time.time() - start_time
        
        logger.info(f"Response generated in {response_time:.2f} seconds")
        
        return jsonify({
            'answer': answer,
            'timestamp': datetime.now().isoformat(),
            'response_time': round(response_time, 2)
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your question. Please try again.'
        }), 500

@app.route('/summarize', methods=['POST'])
def summarize_text():
    """Endpoint to simplify/summarize legal text"""
    global service_ready
    
    if not service_ready:
        return jsonify({
            'error': 'Service is initializing. Please try again in a moment.'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Please provide text to summarize in the request body'
            }), 400
        
        text_to_summarize = data['text'].strip()
        
        if not text_to_summarize:
            return jsonify({
                'error': 'Text cannot be empty'
            }), 400
        
        # Get simplified version
        summary = legal_chatbot.summarize_text(text_to_summarize)
        
        return jsonify({
            'summary': summary,
            'original_length': len(text_to_summarize),
            'summary_length': len(summary),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        return jsonify({
            'error': 'An error occurred while summarizing the text. Please try again.'
        }), 500

@app.route('/clear-history', methods=['POST'])
def clear_conversation_history():
    """Clear conversation history"""
    try:
        legal_chatbot.clear_history()
        return jsonify({
            'message': 'Conversation history cleared successfully',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return jsonify({
            'error': 'An error occurred while clearing history'
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get detailed service status"""
    global model
    model_name = "Not initialized"
    if model and hasattr(model, 'model_name'):
        model_name = model.model_name
    elif model:
        model_name = "Initialized (unknown version)"
        
    return jsonify({
        'service': 'Legal AI Chatbot',
        'status': 'ready' if service_ready else 'initializing',
        'model': model_name,
        'conversation_history_length': len(legal_chatbot.conversation_history),
        'max_history': legal_chatbot.max_history,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/health - GET - Health check',
            '/ask - POST - Ask legal questions',
            '/summarize - POST - Simplify legal text',
            '/clear-history - POST - Clear conversation history',
            '/status - GET - Get service status'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'Please try again later or contact support'
    }), 500

if __name__ == '__main__':
    print("🏛️ Starting Legal AI Chatbot Backend...")
    print("=" * 50)
    print("📋 Available Endpoints:")
    print("  • GET  /health      - Health check")
    print("  • POST /ask         - Ask legal questions")
    print("  • POST /summarize   - Simplify legal text")
    print("  • POST /clear-history - Clear conversation")
    print("  • GET  /status      - Service status")
    print("=" * 50)
    
    # Initialize service on startup
    if initialize_service():
        print("✅ Service initialized successfully!")
    else:
        print("⚠️ Service initialization failed, will retry on first request")
    
    print("🚀 Server starting on http://127.0.0.1:5000")
    print("💡 Make sure your frontend is configured to use this URL")
    print("=" * 50)
    
    # Run the Flask app
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,  # Set to False in production
        threaded=True
    )