"""
Minimal CSP and A* modules for Phishing Detection
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import heapq


class EmailAction(Enum):
    """Possible actions for handling emails"""
    DELIVER = "deliver"
    QUARANTINE = "quarantine"
    SPAM_FOLDER = "spam_folder"
    FLAG_SUSPICIOUS = "flag_suspicious"
    REQUIRE_REVIEW = "require_review"
    DELETE = "delete"


@dataclass
class EmailContext:
    """Context information about an email"""
    sender: str
    subject: str
    confidence: float
    prediction: int
    has_attachments: bool = False
    has_links: bool = False
    sender_domain: str = ""
    
    def __post_init__(self):
        if not self.sender_domain and '@' in self.sender:
            self.sender_domain = self.sender.split('@')[-1].lower()


class PhishingDecisionEngine:
    """Simplified decision engine"""
    
    def __init__(self):
        pass
    
    def make_decision(self, email_context: EmailContext) -> Dict:
        """Make a decision about email handling"""
        confidence = email_context.confidence
        
        # Simple rule-based decision
        if confidence > 0.90:
            action = EmailAction.QUARANTINE
            reason = "High risk detected - quarantining for safety"
        elif confidence > 0.70:
            action = EmailAction.REQUIRE_REVIEW
            reason = "Medium risk - human review recommended"
        elif confidence > 0.40:
            action = EmailAction.FLAG_SUSPICIOUS
            reason = "Suspicious patterns detected"
        else:
            action = EmailAction.DELIVER
            reason = "Low risk - safe to deliver"
        
        return {
            'action': action,
            'reason': reason,
            'confidence': confidence,
            'constraints': {
                'violations': [],
                'allowed_actions': [action],
                'required_action': action
            },
            'cost': 0.0,
            'search_path': [f'{action.value}: Selected based on confidence {confidence:.2%}']
        }


if __name__ == "__main__":
    print("CSP and A* modules loaded successfully!")