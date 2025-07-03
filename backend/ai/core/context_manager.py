"""
Conversation Context Manager for FinancialHub AI
Manages user session context and resolves follow-up questions
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
from django.core.cache import cache
from django.conf import settings

logger = logging.getLogger(__name__)


@dataclass
class QueryContext:
    """Represents a parsed query with its context"""
    user_id: str
    organization_id: str
    original_query: str
    parsed_intent: Dict[str, Any]
    confidence_score: float
    timestamp: datetime
    financial_data: Optional[Dict[str, Any]] = None
    response_summary: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryContext':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class ConversationContextManager:
    """
    Manages conversation context for resolving follow-up questions
    """
    
    def __init__(self, cache_timeout: int = 3600):
        """
        Initialize the context manager
        
        Args:
            cache_timeout: Time in seconds to keep context in cache
        """
        self.cache_timeout = cache_timeout
        self.cache_prefix = "ai_context"
    
    def _get_cache_key(self, user_id: str, organization_id: str) -> str:
        """Generate cache key for user context"""
        return f"{self.cache_prefix}:{organization_id}:{user_id}"
    
    def store_context(self, 
                     user_id: str, 
                     organization_id: str, 
                     query: str, 
                     parsed_intent: Dict[str, Any],
                     confidence_score: float,
                     financial_data: Optional[Dict[str, Any]] = None,
                     response_summary: Optional[str] = None) -> None:
        """
        Store the current query context for a user
        
        Args:
            user_id: User identifier
            organization_id: Organization identifier
            query: Original user query
            parsed_intent: Parsed intent from FinancialQueryParser
            confidence_score: Confidence score of the parsing
            financial_data: Financial data used in the response
            response_summary: Summary of the AI response
        """
        try:
            context = QueryContext(
                user_id=user_id,
                organization_id=organization_id,
                original_query=query,
                parsed_intent=parsed_intent,
                confidence_score=confidence_score,
                timestamp=datetime.now(),
                financial_data=financial_data,
                response_summary=response_summary
            )
            
            cache_key = self._get_cache_key(user_id, organization_id)
            cache.set(cache_key, context.to_dict(), self.cache_timeout)
            
            logger.info(f"Stored context for user {user_id} in org {organization_id}")
            
        except Exception as e:
            logger.error(f"Error storing context: {e}")
    
    def get_context(self, user_id: str, organization_id: str) -> Optional[QueryContext]:
        """
        Retrieve the current context for a user
        
        Args:
            user_id: User identifier
            organization_id: Organization identifier
            
        Returns:
            QueryContext object or None if not found
        """
        try:
            cache_key = self._get_cache_key(user_id, organization_id)
            context_data = cache.get(cache_key)
            
            if context_data:
                return QueryContext.from_dict(context_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return None
    
    def resolve_follow_up(self, 
                         user_id: str, 
                         organization_id: str, 
                         current_query: str) -> Optional[Dict[str, Any]]:
        """
        Resolve follow-up questions by analyzing context
        
        Args:
            user_id: User identifier
            organization_id: Organization identifier
            current_query: Current user query
            
        Returns:
            Resolved context or None if not a follow-up
        """
        try:
            context = self.get_context(user_id, organization_id)
            if not context:
                return None
            
            # Check if this is a follow-up question
            follow_up_indicators = [
                "last month", "previous month", "last week", "yesterday",
                "el mes pasado", "la semana pasada", "ayer",
                "what about", "how about", "and", "also",
                "qué tal", "cómo", "también", "además"
            ]
            
            query_lower = current_query.lower()
            is_follow_up = any(indicator in query_lower for indicator in follow_up_indicators)
            
            if not is_follow_up:
                return None
            
            # Resolve time references
            resolved_context = self._resolve_time_references(
                context, current_query
            )
            
            return resolved_context
            
        except Exception as e:
            logger.error(f"Error resolving follow-up: {e}")
            return None
    
    def _resolve_time_references(self, 
                                context: QueryContext, 
                                current_query: str) -> Dict[str, Any]:
        """
        Resolve time references in follow-up questions
        
        Args:
            context: Previous query context
            current_query: Current query with time references
            
        Returns:
            Resolved context with updated time parameters
        """
        query_lower = current_query.lower()
        now = datetime.now()
        
        # Time reference mapping
        time_mappings = {
            "last month": now - timedelta(days=30),
            "previous month": now - timedelta(days=30),
            "last week": now - timedelta(days=7),
            "yesterday": now - timedelta(days=1),
            "el mes pasado": now - timedelta(days=30),
            "la semana pasada": now - timedelta(days=7),
            "ayer": now - timedelta(days=1)
        }
        
        resolved_context = context.parsed_intent.copy()
        
        for time_ref, target_date in time_mappings.items():
            if time_ref in query_lower:
                # Update date parameters in the intent
                if 'date_range' in resolved_context:
                    resolved_context['date_range'] = {
                        'start_date': target_date.strftime('%Y-%m-%d'),
                        'end_date': now.strftime('%Y-%m-%d')
                    }
                elif 'target_date' in resolved_context:
                    resolved_context['target_date'] = target_date.strftime('%Y-%m-%d')
                
                break
        
        return {
            'original_context': context.to_dict() if hasattr(context, 'to_dict') else str(context),
            'resolved_intent': resolved_context,
            'is_follow_up': True
        }
    
    def clear_context(self, user_id: str, organization_id: str) -> None:
        """
        Clear the context for a user
        
        Args:
            user_id: User identifier
            organization_id: Organization identifier
        """
        try:
            cache_key = self._get_cache_key(user_id, organization_id)
            cache.delete(cache_key)
            logger.info(f"Cleared context for user {user_id} in org {organization_id}")
            
        except Exception as e:
            logger.error(f"Error clearing context: {e}")
    
    def get_context_history(self, 
                           user_id: str, 
                           organization_id: str, 
                           limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent context history for a user
        
        Args:
            user_id: User identifier
            organization_id: Organization identifier
            limit: Maximum number of contexts to return
            
        Returns:
            List of recent QueryContext dictionaries
        """
        # This would typically use a database table for persistence
        # For now, we'll return the current context if available
        context = self.get_context(user_id, organization_id)
        return [context.to_dict()] if context else [] 