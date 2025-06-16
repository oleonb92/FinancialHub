"""
AI services for FinancialHub.
"""
from django.conf import settings
from django.utils import timezone
from .models import AIInteraction, AIInsight, AIPrediction
from .ml.classifiers.transaction import TransactionClassifier
from .ml.predictors.expense import ExpensePredictor
from .ml.analyzers.behavior import BehaviorAnalyzer
from transactions.models import Transaction
import json
import logging

logger = logging.getLogger('ai.services')

class AIService:
    """
    Service for handling AI-related operations.
    """
    
    def __init__(self):
        """
        Initialize AI service with ML models.
        """
        self.transaction_classifier = TransactionClassifier()
        self.expense_predictor = ExpensePredictor()
        self.behavior_analyzer = BehaviorAnalyzer()
        
        # Load trained models if available
        try:
            self.transaction_classifier.load()
            self.expense_predictor.load()
            self.behavior_analyzer.load()
        except Exception as e:
            logger.warning(f"Could not load trained models: {str(e)}")
    
    def process_query(self, user, query, context=None, interaction_type='general'):
        """
        Process a user query and generate an AI response.
        
        Args:
            user: User object
            query: User query string
            context: Additional context data
            interaction_type: Type of interaction
            
        Returns:
            dict: Response data
        """
        try:
            # Create interaction record
            interaction = AIInteraction.objects.create(
                user=user,
                type=interaction_type,
                query=query,
                context=context or {}
            )
            
            # Process based on interaction type
            if interaction_type == 'transaction':
                response = self._process_transaction_query(query, context)
            elif interaction_type == 'budget':
                response = self._process_budget_query(query, context)
            elif interaction_type == 'prediction':
                response = self._process_prediction_query(query, context)
            else:
                response = self._process_general_query(query, context)
            
            # Update interaction with response
            interaction.response = response
            interaction.confidence_score = self._calculate_confidence_score(response)
            interaction.save()
            
            # Generate insights if applicable
            if interaction_type in ['transaction', 'budget', 'goal']:
                self._generate_insights(user, interaction)
            
            return {
                'response': response,
                'confidence_score': interaction.confidence_score,
                'interaction_id': interaction.id
            }
            
        except Exception as e:
            logger.error(f"Error processing AI query: {str(e)}")
            return {
                'error': 'Unable to process your request at this time',
                'details': str(e)
            }
    
    def analyze_transaction(self, transaction):
        """
        Analyze a transaction using ML models.
        
        Args:
            transaction: Transaction object
            
        Returns:
            dict: Analysis results
        """
        try:
            # Get category prediction
            category_id, confidence = self.transaction_classifier.predict(transaction)
            
            # Update transaction
            transaction.ai_analyzed = True
            transaction.ai_confidence = confidence
            transaction.ai_category_suggestion_id = category_id
            
            # Get behavior analysis
            behavior_analysis = self.behavior_analyzer.analyze_spending_patterns([transaction])
            
            return {
                'category_suggestion': category_id,
                'confidence_score': confidence,
                'behavior_analysis': behavior_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing transaction: {str(e)}")
            raise
    
    def predict_expenses(self, user, start_date, days=30):
        """
        Predict future expenses.
        
        Args:
            user: User object
            start_date: Start date for prediction
            days: Number of days to predict
            
        Returns:
            dict: Prediction results
        """
        try:
            # Get historical transactions
            transactions = Transaction.objects.filter(
                user=user,
                date__lt=start_date
            ).order_by('-date')[:1000]
            
            # Train predictor if needed
            if not self.expense_predictor.model:
                self.expense_predictor.train(transactions)
            
            # Make predictions
            predictions = self.expense_predictor.predict_sequence(start_date, days)
            
            # Create prediction record
            prediction = AIPrediction.objects.create(
                user=user,
                type='expense',
                prediction=predictions.to_dict('records'),
                confidence_score=0.8,  # Placeholder
                prediction_date=start_date
            )
            
            return {
                'predictions': predictions.to_dict('records'),
                'prediction_id': prediction.id
            }
            
        except Exception as e:
            logger.error(f"Error predicting expenses: {str(e)}")
            raise
    
    def analyze_spending_patterns(self, user):
        """
        Analyze user's spending patterns.
        
        Args:
            user: User object
            
        Returns:
            dict: Pattern analysis results
        """
        try:
            # Get recent transactions
            transactions = Transaction.objects.filter(
                user=user
            ).order_by('-date')[:1000]
            
            # Analyze patterns
            patterns = self.behavior_analyzer.analyze_spending_patterns(transactions)
            
            # Create insight record
            insight = AIInsight.objects.create(
                user=user,
                type='spending',
                title='Spending Pattern Analysis',
                description=json.dumps(patterns),
                data=patterns
            )
            
            return {
                'patterns': patterns,
                'insight_id': insight.id
            }
            
        except Exception as e:
            logger.error(f"Error analyzing spending patterns: {str(e)}")
            raise
    
    def _process_transaction_query(self, query, context):
        """
        Process a transaction-related query.
        """
        # Implement transaction query processing
        return "Transaction analysis completed"
    
    def _process_budget_query(self, query, context):
        """
        Process a budget-related query.
        """
        # Implement budget query processing
        return "Budget analysis completed"
    
    def _process_prediction_query(self, query, context):
        """
        Process a prediction-related query.
        """
        # Implement prediction query processing
        return "Prediction analysis completed"
    
    def _process_general_query(self, query, context):
        """
        Process a general query.
        """
        # Implement general query processing
        return "General analysis completed"
    
    def _calculate_confidence_score(self, response):
        """
        Calculate confidence score for a response.
        """
        # Implement confidence score calculation
        return 0.8
    
    def _generate_insights(self, user, interaction):
        """
        Generate insights based on an interaction.
        """
        try:
            # Create insight record
            AIInsight.objects.create(
                user=user,
                type=interaction.type,
                title=f"Insight from {interaction.type} analysis",
                description=interaction.response,
                data={
                    'interaction_id': interaction.id,
                    'query': interaction.query,
                    'response': interaction.response
                }
            )
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}") 