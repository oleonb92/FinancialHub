from django.db import models
from django.conf import settings
# Translation handled by TranslationService

class AIInteraction(models.Model):
    INTERACTION_TYPES = (
        ('query', 'Query'),
        ('analysis', 'Analysis'),
        ('prediction', 'Prediction'),
        ('recommendation', 'Recommendation'),
        ('alert', 'Alert'),
    )

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='ai_interactions')
    organization = models.ForeignKey('organizations.Organization', on_delete=models.CASCADE, related_name='ai_interactions', null=True, blank=True)
    type = models.CharField(max_length=20, choices=INTERACTION_TYPES)
    query = models.TextField()
    response = models.TextField()
    context = models.JSONField(default=dict, blank=True)  # Additional context data
    created_at = models.DateTimeField(auto_now_add=True)
    confidence_score = models.FloatField(null=True, blank=True)
    feedback = models.BooleanField(null=True, blank=True)  # User feedback on response
    feedback_comment = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'type']),
            models.Index(fields=['created_at']),
            models.Index(fields=['organization']),
        ]

    def __str__(self):
        return f"AI {self.get_type_display()} for {self.user.username}"

class AIInsight(models.Model):
    INSIGHT_TYPES = (
        ('budget', 'Budget'),
        ('spending', 'Spending'),
        ('saving', 'Saving'),
        ('investment', 'Investment'),
        ('risk', 'Risk'),
        ('opportunity', 'Opportunity'),
    )

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='ai_insights')
    organization = models.ForeignKey('organizations.Organization', on_delete=models.CASCADE, related_name='ai_insights', null=True, blank=True)
    type = models.CharField(max_length=20, choices=INSIGHT_TYPES)
    title = models.CharField(max_length=255)
    description = models.TextField()
    data = models.JSONField(default=dict, blank=True)  # Supporting data
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    action_taken = models.BooleanField(default=False)
    action_description = models.TextField(blank=True, null=True)
    impact_score = models.FloatField(null=True, blank=True)  # Estimated impact score

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'type']),
            models.Index(fields=['is_read']),
            models.Index(fields=['organization']),
        ]

    def __str__(self):
        return f"AI Insight: {self.title} for {self.user.username}"

class AIPrediction(models.Model):
    PREDICTION_TYPES = (
        ('budget', 'Budget'),
        ('cash_flow', 'Cash Flow'),
        ('spending', 'Spending'),
        ('saving', 'Saving'),
        ('risk', 'Risk'),
    )

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='ai_predictions')
    organization = models.ForeignKey('organizations.Organization', on_delete=models.CASCADE, related_name='ai_predictions', null=True, blank=True)
    type = models.CharField(max_length=20, choices=PREDICTION_TYPES)
    prediction = models.JSONField()  # Structured prediction data
    confidence_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    prediction_date = models.DateField()  # Date for which prediction is made
    actual_result = models.JSONField(null=True, blank=True)  # Actual result when available
    accuracy_score = models.FloatField(null=True, blank=True)  # Accuracy score when actual result is available

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'type']),
            models.Index(fields=['prediction_date']),
            models.Index(fields=['organization']),
        ]

    def __str__(self):
        return f"AI Prediction: {self.get_type_display()} for {self.user.username} on {self.prediction_date}"

class AIQueryLog(models.Model):
    """
    Log model for tracking AI query interactions and performance.
    
    This model stores comprehensive information about user queries, intents,
    confidence scores, and processing metadata for analysis and fine-tuning.
    """
    
    # Basic query information
    query_text = models.TextField(help_text="Original user query text")
    language = models.CharField(max_length=10, default='en', help_text="Detected language of the query")
    
    # Intent classification results
    detected_intent = models.CharField(max_length=100, help_text="Primary detected intent")
    confidence_score = models.FloatField(help_text="Confidence score for the intent")
    all_intent_scores = models.JSONField(default=dict, help_text="All intent scores and probabilities")
    
    # Entity extraction results
    extracted_entities = models.JSONField(default=dict, help_text="Extracted entities and their values")
    extracted_dates = models.JSONField(default=list, help_text="Extracted date information")
    extracted_amounts = models.JSONField(default=list, help_text="Extracted monetary amounts")
    extracted_categories = models.JSONField(default=list, help_text="Extracted expense categories")
    extracted_accounts = models.JSONField(default=list, help_text="Extracted account information")
    
    # Processing metadata
    processing_time = models.FloatField(help_text="Time taken to process the query (seconds)")
    complexity_score = models.FloatField(default=0.0, help_text="Query complexity score")
    entity_count = models.IntegerField(default=0, help_text="Number of entities extracted")
    date_count = models.IntegerField(default=0, help_text="Number of dates extracted")
    amount_count = models.IntegerField(default=0, help_text="Number of amounts extracted")
    
    # System information
    spacy_used = models.BooleanField(default=False, help_text="Whether spaCy was used for processing")
    dateparser_used = models.BooleanField(default=False, help_text="Whether dateparser was used")
    model_version = models.CharField(max_length=50, default='1.0', help_text="Version of the AI model used")
    
    # Response information
    response_generated = models.BooleanField(default=False, help_text="Whether a response was generated")
    response_template_id = models.CharField(max_length=100, blank=True, null=True, help_text="Template ID used for response")
    response_quality_score = models.FloatField(null=True, blank=True, help_text="Quality score of the generated response")
    
    # User and session information
    user = models.ForeignKey('accounts.User', on_delete=models.CASCADE, null=True, blank=True, help_text="User who made the query")
    organization = models.ForeignKey('organizations.Organization', on_delete=models.CASCADE, null=True, blank=True, help_text="Organization context")
    session_id = models.CharField(max_length=100, blank=True, null=True, help_text="Session identifier")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, help_text="When the query was processed")
    updated_at = models.DateTimeField(auto_now=True, help_text="When the log was last updated")
    
    # Error tracking
    processing_error = models.TextField(blank=True, null=True, help_text="Error message if processing failed")
    error_type = models.CharField(max_length=100, blank=True, null=True, help_text="Type of error that occurred")
    
    # Analytics flags
    is_training_data = models.BooleanField(default=False, help_text="Whether this query should be used for training")
    is_anomaly = models.BooleanField(default=False, help_text="Whether this query was flagged as anomalous")
    needs_human_review = models.BooleanField(default=False, help_text="Whether this query needs human review")
    
    # Additional metadata
    metadata = models.JSONField(default=dict, help_text="Additional processing metadata")
    
    class Meta:
        db_table = 'ai_query_logs'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['created_at']),
            models.Index(fields=['detected_intent']),
            models.Index(fields=['user']),
            models.Index(fields=['organization']),
            models.Index(fields=['language']),
            models.Index(fields=['confidence_score']),
            models.Index(fields=['is_training_data']),
            models.Index(fields=['is_anomaly']),
        ]
        verbose_name = "AI Query Log"
        verbose_name_plural = "AI Query Logs"
    
    def __str__(self):
        return f"Query: {self.query_text[:50]}... ({self.detected_intent})"
    
    @property
    def query_length(self):
        """Return the length of the query text."""
        return len(self.query_text)
    
    @property
    def has_entities(self):
        """Check if any entities were extracted."""
        return bool(self.extracted_entities)
    
    @property
    def has_dates(self):
        """Check if any dates were extracted."""
        return bool(self.extracted_dates)
    
    @property
    def has_amounts(self):
        """Check if any amounts were extracted."""
        return bool(self.extracted_amounts)
    
    @property
    def confidence_level(self):
        """Return confidence level as a string."""
        if self.confidence_score >= 0.9:
            return "Very High"
        elif self.confidence_score >= 0.7:
            return "High"
        elif self.confidence_score >= 0.5:
            return "Medium"
        elif self.confidence_score >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    @property
    def complexity_level(self):
        """Return complexity level as a string."""
        if self.complexity_score >= 0.8:
            return "Very Complex"
        elif self.complexity_score >= 0.6:
            return "Complex"
        elif self.complexity_score >= 0.4:
            return "Moderate"
        elif self.complexity_score >= 0.2:
            return "Simple"
        else:
            return "Very Simple"
    
    def get_primary_entity(self):
        """Get the primary entity from extracted entities."""
        if not self.extracted_entities:
            return None
        
        # Priority order for entities
        priority_entities = ['financial_entity', 'time_period', 'comparison_type']
        
        for entity_type in priority_entities:
            if entity_type in self.extracted_entities:
                return {
                    'type': entity_type,
                    'value': self.extracted_entities[entity_type]
                }
        
        # Return first available entity
        first_key = list(self.extracted_entities.keys())[0]
        return {
            'type': first_key,
            'value': self.extracted_entities[first_key]
        }
    
    def get_largest_amount(self):
        """Get the largest monetary amount from extracted amounts."""
        if not self.extracted_amounts:
            return None
        
        amounts = [amount.get('value', 0) for amount in self.extracted_amounts if amount.get('value')]
        if amounts:
            return max(amounts)
        return None
    
    def get_primary_date(self):
        """Get the primary date from extracted dates."""
        if not self.extracted_dates:
            return None
        
        # Return the first date with highest confidence
        sorted_dates = sorted(self.extracted_dates, key=lambda x: x.get('confidence', 0), reverse=True)
        return sorted_dates[0] if sorted_dates else None
    
    def mark_for_training(self):
        """Mark this query for use in training data."""
        self.is_training_data = True
        self.save(update_fields=['is_training_data', 'updated_at'])
    
    def mark_as_anomaly(self):
        """Mark this query as anomalous."""
        self.is_anomaly = True
        self.save(update_fields=['is_anomaly', 'updated_at'])
    
    def mark_for_review(self):
        """Mark this query for human review."""
        self.needs_human_review = True
        self.save(update_fields=['needs_human_review', 'updated_at'])
    
    def update_response_info(self, template_id=None, quality_score=None):
        """Update response information."""
        self.response_generated = True
        if template_id:
            self.response_template_id = template_id
        if quality_score is not None:
            self.response_quality_score = quality_score
        self.save(update_fields=['response_generated', 'response_template_id', 'response_quality_score', 'updated_at'])
    
    @classmethod
    def get_analytics_summary(cls, organization=None, start_date=None, end_date=None):
        """Get analytics summary for the specified period."""
        queryset = cls.objects.all()
        
        if organization:
            queryset = queryset.filter(organization=organization)
        
        if start_date:
            queryset = queryset.filter(created_at__gte=start_date)
        
        if end_date:
            queryset = queryset.filter(created_at__lte=end_date)
        
        total_queries = queryset.count()
        
        if total_queries == 0:
            return {
                'total_queries': 0,
                'avg_confidence': 0,
                'avg_processing_time': 0,
                'intent_distribution': {},
                'language_distribution': {},
                'error_rate': 0,
                'training_data_count': 0,
                'anomaly_count': 0
            }
        
        # Calculate averages
        avg_confidence = queryset.aggregate(avg=models.Avg('confidence_score'))['avg'] or 0
        avg_processing_time = queryset.aggregate(avg=models.Avg('processing_time'))['avg'] or 0
        
        # Intent distribution
        intent_distribution = queryset.values('detected_intent').annotate(
            count=models.Count('id')
        ).order_by('-count')
        
        # Language distribution
        language_distribution = queryset.values('language').annotate(
            count=models.Count('id')
        ).order_by('-count')
        
        # Error rate
        error_count = queryset.filter(processing_error__isnull=False).count()
        error_rate = (error_count / total_queries) * 100
        
        # Training data and anomalies
        training_data_count = queryset.filter(is_training_data=True).count()
        anomaly_count = queryset.filter(is_anomaly=True).count()
        
        return {
            'total_queries': total_queries,
            'avg_confidence': round(avg_confidence, 3),
            'avg_processing_time': round(avg_processing_time, 3),
            'intent_distribution': {item['detected_intent']: item['count'] for item in intent_distribution},
            'language_distribution': {item['language']: item['count'] for item in language_distribution},
            'error_rate': round(error_rate, 2),
            'training_data_count': training_data_count,
            'anomaly_count': anomaly_count
        }
    
    @classmethod
    def get_training_data(cls, organization=None, min_confidence=0.7, limit=1000):
        """Get queries suitable for training data."""
        queryset = cls.objects.filter(
            is_training_data=True,
            confidence_score__gte=min_confidence,
            processing_error__isnull=True
        )
        
        if organization:
            queryset = queryset.filter(organization=organization)
        
        return queryset.order_by('-created_at')[:limit]
    
    @classmethod
    def get_anomalies(cls, organization=None, limit=100):
        """Get anomalous queries for review."""
        queryset = cls.objects.filter(is_anomaly=True)
        
        if organization:
            queryset = queryset.filter(organization=organization)
        
        return queryset.order_by('-created_at')[:limit]
    
    @classmethod
    def get_low_confidence_queries(cls, organization=None, max_confidence=0.5, limit=100):
        """Get queries with low confidence for review."""
        queryset = cls.objects.filter(confidence_score__lte=max_confidence)
        
        if organization:
            queryset = queryset.filter(organization=organization)
        
        return queryset.order_by('-created_at')[:limit]


class AIQueryLogManager(models.Manager):
    """Custom manager for AIQueryLog with additional query methods."""
    
    def create_from_parsed_intent(self, query_text, parsed_intent, user=None, organization=None, **kwargs):
        """Create a log entry from a parsed intent object."""
        from .core.enhanced_query_parser import EnhancedParsedIntent
        
        if isinstance(parsed_intent, EnhancedParsedIntent):
            return self.create(
                query_text=query_text,
                language=parsed_intent.language,
                detected_intent=parsed_intent.intent_type,
                confidence_score=parsed_intent.confidence_score,
                all_intent_scores=parsed_intent.all_scores,
                extracted_entities=parsed_intent.entities,
                extracted_dates=parsed_intent.extracted_dates,
                extracted_amounts=parsed_intent.extracted_amounts,
                extracted_categories=parsed_intent.extracted_categories,
                extracted_accounts=parsed_intent.extracted_accounts,
                processing_time=parsed_intent.processing_time,
                complexity_score=parsed_intent.metadata.get('complexity_score', 0),
                entity_count=parsed_intent.metadata.get('entity_count', 0),
                date_count=parsed_intent.metadata.get('date_count', 0),
                amount_count=parsed_intent.metadata.get('amount_count', 0),
                spacy_used=parsed_intent.metadata.get('spacy_used', False),
                dateparser_used=parsed_intent.metadata.get('dateparser_used', False),
                user=user,
                organization=organization,
                metadata=parsed_intent.metadata,
                **kwargs
            )
        else:
            # Fallback for basic parsed intent
            return self.create(
                query_text=query_text,
                detected_intent=getattr(parsed_intent, 'intent_type', 'unknown'),
                confidence_score=getattr(parsed_intent, 'confidence_score', 0.0),
                user=user,
                organization=organization,
                **kwargs
            )
    
    def get_performance_metrics(self, organization=None, days=30):
        """Get performance metrics for the specified period."""
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return AIQueryLog.get_analytics_summary(organization, start_date, end_date)
    
    def get_intent_accuracy(self, organization=None, days=30):
        """Get intent classification accuracy metrics."""
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        queryset = self.filter(created_at__range=[start_date, end_date])
        if organization:
            queryset = queryset.filter(organization=organization)
        
        total_queries = queryset.count()
        if total_queries == 0:
            return {}
        
        # High confidence queries (confidence >= 0.8)
        high_confidence = queryset.filter(confidence_score__gte=0.8).count()
        
        # Medium confidence queries (0.5 <= confidence < 0.8)
        medium_confidence = queryset.filter(confidence_score__gte=0.5, confidence_score__lt=0.8).count()
        
        # Low confidence queries (confidence < 0.5)
        low_confidence = queryset.filter(confidence_score__lt=0.5).count()
        
        return {
            'total_queries': total_queries,
            'high_confidence_rate': round((high_confidence / total_queries) * 100, 2),
            'medium_confidence_rate': round((medium_confidence / total_queries) * 100, 2),
            'low_confidence_rate': round((low_confidence / total_queries) * 100, 2),
            'avg_confidence': round(queryset.aggregate(avg=models.Avg('confidence_score'))['avg'] or 0, 3)
        }


# Add the custom manager to the model
AIQueryLog.objects = AIQueryLogManager() 