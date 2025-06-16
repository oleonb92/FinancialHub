# backend/financialhub/middleware.py

import traceback
from django.http import HttpResponseServerError
import logging
logger = logging.getLogger(__name__)

class Show500ErrorMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            return self.get_response(request)
        except Exception as e:
            logger.error("❌ Exception caught in middleware:", exc_info=True)
            return HttpResponseServerError(f"""
                &lt;html&gt;
                    &lt;head&gt;&lt;title&gt;500 Internal Server Error&lt;/title&gt;&lt;/head&gt;
                    &lt;body style="font-family: monospace; background-color: #f9f9f9; padding: 2em;"&gt;
                        &lt;h1 style="color: #e74c3c;"&gt;500 Internal Server Error&lt;/h1&gt;
                        &lt;p&gt;An unexpected error occurred. Here's the stack trace:&lt;/p&gt;
                        &lt;pre style="background: #eee; padding: 1em; border: 1px solid #ccc;"&gt;{traceback.format_exc()}&lt;/pre&gt;
                    &lt;/body&gt;
                &lt;/html&gt;
            """, content_type="text/html")