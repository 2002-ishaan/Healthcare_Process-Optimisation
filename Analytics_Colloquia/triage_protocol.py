# =============================================================================
# TRIAGE LEAD OPERATIONAL PROTOCOL
# NTH-ED Decision Support System
# =============================================================================

"""
This module provides decision rules and action recommendations for the Triage Lead
based on ML model predictions.

Usage:
    from triage_protocol import TriageProtocol
    
    protocol = TriageProtocol()
    recommendation = protocol.get_recommendation(admission_prob=0.72, lwbs_prob=0.05)
    print(recommendation)
"""

# =============================================================================
# THRESHOLDS (From Model Results)
# =============================================================================

THRESHOLDS = {
    'admission': {
        'high': 0.68,      # ≥68% = HIGH risk
        'medium': 0.50,    # 50-67% = MEDIUM risk
    },
    'lwbs': {
        'top_5': 0.026,    # Top 5% threshold
        'top_10': 0.015,   # Top 10% threshold (operational)
        'top_15': 0.010,   # Top 15% threshold
    }
}

# =============================================================================
# MODEL PERFORMANCE (For Reference)
# =============================================================================

MODEL_PERFORMANCE = {
    'admission': {
        'auc': 0.8495,
        'precision': 0.4526,  # 45% of flags correct
        'recall': 0.6225,     # 62% of admissions caught
        'threshold': 0.676,
    },
    'lwbs': {
        'auc': 0.85,
        'top_5_lift': 7.7,    # 7.7x better than random
        'top_5_capture': 0.383,  # Catches 38% of LWBS
        'top_10_capture': 0.532, # Catches 53% of LWBS
    }
}

# =============================================================================
# ACTION DEFINITIONS
# =============================================================================

ADMISSION_ACTIONS = {
    'HIGH': [
        "🛏️ Start bed search IMMEDIATELY",
        "📞 Notify admitting team",
        "🩺 Request early consult if needed",
        "📋 Document admission likelihood"
    ],
    'MEDIUM': [
        "👁️ Monitor for deterioration",
        "📝 Prepare admission paperwork",
        "🔔 Alert charge nurse"
    ],
    'LOW': [
        "✓ Standard care pathway",
        "📊 Re-assess if condition changes"
    ]
}

LWBS_ACTIONS = {
    'TOP_5': [
        "⏰ Check on patient at 15 min",
        "💬 Proactively communicate wait time",
        "🪑 Ensure comfortable waiting area",
        "📱 Offer to update family"
    ],
    'TOP_10': [
        "⏰ Check on patient at 25 min",
        "💬 Update on expected wait",
        "📋 Add to watch list"
    ],
    'TOP_15': [
        "⏰ Check on patient at 30 min",
        "📋 Monitor wait time"
    ],
    'STANDARD': [
        "✓ Standard monitoring"
    ]
}

# =============================================================================
# PRIORITY PATIENT PROFILES
# =============================================================================

HIGH_ADMISSION_RISK_PROFILES = [
    "Elderly (65+) with CTAS 1-2",
    "Ambulance arrival + high acuity",
    "Yellow Zone + high acuity",
    "History of frequent admissions (if known)"
]

HIGH_LWBS_RISK_PROFILES = [
    "CTAS 4-5 during peak hours (10 AM - 10 PM)",
    "Green Zone on weekdays",
    "Young/middle-aged males",
    "Non-ambulance, walk-in arrivals",
    "Weekend + low acuity"
]

# =============================================================================
# TRIAGE PROTOCOL CLASS
# =============================================================================

class TriageProtocol:
    """
    Decision support protocol for Triage Lead.
    
    Converts model probabilities into actionable recommendations.
    """
    
    def __init__(self, thresholds: dict = None):
        self.thresholds = thresholds or THRESHOLDS
    
    # -------------------------------------------------------------------------
    # RISK CLASSIFICATION
    # -------------------------------------------------------------------------
    
    def classify_admission_risk(self, probability: float) -> str:
        """Classify admission risk level."""
        if probability >= self.thresholds['admission']['high']:
            return 'HIGH'
        elif probability >= self.thresholds['admission']['medium']:
            return 'MEDIUM'
        return 'LOW'
    
    def classify_lwbs_risk(self, probability: float) -> str:
        """Classify LWBS risk tier."""
        if probability >= self.thresholds['lwbs']['top_5']:
            return 'TOP_5'
        elif probability >= self.thresholds['lwbs']['top_10']:
            return 'TOP_10'
        elif probability >= self.thresholds['lwbs']['top_15']:
            return 'TOP_15'
        return 'STANDARD'
    
    # -------------------------------------------------------------------------
    # ACTION RECOMMENDATIONS
    # -------------------------------------------------------------------------
    
    def get_admission_actions(self, risk_level: str) -> list:
        """Get recommended actions for admission risk."""
        return ADMISSION_ACTIONS.get(risk_level, ADMISSION_ACTIONS['LOW'])
    
    def get_lwbs_actions(self, risk_tier: str) -> list:
        """Get recommended actions for LWBS risk."""
        return LWBS_ACTIONS.get(risk_tier, LWBS_ACTIONS['STANDARD'])
    
    def get_check_time(self, lwbs_tier: str) -> int:
        """Get recommended check time in minutes."""
        check_times = {
            'TOP_5': 15,
            'TOP_10': 25,
            'TOP_15': 30,
            'STANDARD': 45
        }
        return check_times.get(lwbs_tier, 45)
    
    # -------------------------------------------------------------------------
    # COMBINED RECOMMENDATION
    # -------------------------------------------------------------------------
    
    def get_recommendation(self, admission_prob: float, lwbs_prob: float) -> dict:
        """
        Generate complete recommendation for a patient.
        
        Parameters:
        -----------
        admission_prob : float (0-1)
            Probability of hospital admission from ML model
        lwbs_prob : float (0-1)
            Probability of LWBS from ML model
            
        Returns:
        --------
        dict with risk levels, actions, and priority
        """
        
        # Classify risks
        adm_risk = self.classify_admission_risk(admission_prob)
        lwbs_tier = self.classify_lwbs_risk(lwbs_prob)
        
        # Get actions
        adm_actions = self.get_admission_actions(adm_risk)
        lwbs_actions = self.get_lwbs_actions(lwbs_tier)
        
        # Calculate priority score (for sorting patients)
        priority_score = self._calculate_priority(admission_prob, lwbs_prob, adm_risk, lwbs_tier)
        
        # Determine overall urgency
        urgency = self._determine_urgency(adm_risk, lwbs_tier)
        
        return {
            'admission': {
                'probability': admission_prob,
                'risk_level': adm_risk,
                'actions': adm_actions
            },
            'lwbs': {
                'probability': lwbs_prob,
                'risk_tier': lwbs_tier,
                'check_time_min': self.get_check_time(lwbs_tier),
                'actions': lwbs_actions
            },
            'priority_score': priority_score,
            'urgency': urgency,
            'summary': self._generate_summary(adm_risk, lwbs_tier)
        }
    
    def _calculate_priority(self, adm_prob: float, lwbs_prob: float, 
                           adm_risk: str, lwbs_tier: str) -> float:
        """Calculate priority score for patient queue ordering."""
        
        # Base score from probabilities
        score = (adm_prob * 0.6) + (lwbs_prob * 0.4)
        
        # Boost for high risk
        if adm_risk == 'HIGH':
            score += 0.2
        if lwbs_tier in ['TOP_5', 'TOP_10']:
            score += 0.1
            
        return min(score, 1.0)
    
    def _determine_urgency(self, adm_risk: str, lwbs_tier: str) -> str:
        """Determine overall urgency level."""
        
        if adm_risk == 'HIGH' and lwbs_tier in ['TOP_5', 'TOP_10']:
            return 'CRITICAL'
        elif adm_risk == 'HIGH' or lwbs_tier == 'TOP_5':
            return 'HIGH'
        elif adm_risk == 'MEDIUM' or lwbs_tier == 'TOP_10':
            return 'MEDIUM'
        return 'STANDARD'
    
    def _generate_summary(self, adm_risk: str, lwbs_tier: str) -> str:
        """Generate one-line summary for dashboard display."""
        
        summaries = {
            ('HIGH', 'TOP_5'): "🚨 CRITICAL: High admission + LWBS risk. Immediate action required.",
            ('HIGH', 'TOP_10'): "🔴 HIGH: Likely admission. Start bed search. Monitor for LWBS.",
            ('HIGH', 'TOP_15'): "🔴 HIGH: Likely admission. Start bed search.",
            ('HIGH', 'STANDARD'): "🔴 HIGH: Likely admission. Start bed search.",
            ('MEDIUM', 'TOP_5'): "🟠 MEDIUM: Possible admission. HIGH LWBS risk - check at 15 min.",
            ('MEDIUM', 'TOP_10'): "🟠 MEDIUM: Monitor for admission. Check patient at 25 min.",
            ('MEDIUM', 'TOP_15'): "🟡 MEDIUM: Monitor for admission and LWBS.",
            ('MEDIUM', 'STANDARD'): "🟡 MEDIUM: Monitor for possible admission.",
            ('LOW', 'TOP_5'): "⚠️ LWBS RISK: Low admission risk but check at 15 min.",
            ('LOW', 'TOP_10'): "📋 Watch list: Check patient at 25 min.",
            ('LOW', 'TOP_15'): "✓ Standard care. Brief check at 30 min.",
            ('LOW', 'STANDARD'): "✓ Standard care pathway."
        }
        
        return summaries.get((adm_risk, lwbs_tier), "✓ Standard care pathway.")


# =============================================================================
# BATCH PROCESSING FUNCTIONS
# =============================================================================

def process_patient_batch(patients_df, admission_probs, lwbs_probs):
    """
    Process a batch of patients and generate recommendations.
    
    Parameters:
    -----------
    patients_df : DataFrame with patient info
    admission_probs : array of admission probabilities
    lwbs_probs : array of LWBS probabilities
    
    Returns:
    --------
    DataFrame with recommendations
    """
    
    protocol = TriageProtocol()
    
    results = []
    for i in range(len(patients_df)):
        rec = protocol.get_recommendation(
            admission_prob=admission_probs[i],
            lwbs_prob=lwbs_probs[i]
        )
        
        results.append({
            'patient_id': patients_df.iloc[i].get('case_id', i),
            'admission_risk': rec['admission']['risk_level'],
            'admission_prob': rec['admission']['probability'],
            'lwbs_tier': rec['lwbs']['risk_tier'],
            'lwbs_prob': rec['lwbs']['probability'],
            'check_time_min': rec['lwbs']['check_time_min'],
            'priority_score': rec['priority_score'],
            'urgency': rec['urgency'],
            'summary': rec['summary']
        })
    
    import pandas as pd
    return pd.DataFrame(results).sort_values('priority_score', ascending=False)


def get_shift_summary(recommendations_df):
    """
    Generate shift summary from batch recommendations.
    
    Returns dict with counts by risk level.
    """
    
    return {
        'total_patients': len(recommendations_df),
        'high_admission': (recommendations_df['admission_risk'] == 'HIGH').sum(),
        'medium_admission': (recommendations_df['admission_risk'] == 'MEDIUM').sum(),
        'top_5_lwbs': (recommendations_df['lwbs_tier'] == 'TOP_5').sum(),
        'top_10_lwbs': (recommendations_df['lwbs_tier'].isin(['TOP_5', 'TOP_10'])).sum(),
        'critical_urgency': (recommendations_df['urgency'] == 'CRITICAL').sum(),
        'high_urgency': (recommendations_df['urgency'] == 'HIGH').sum()
    }


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def print_recommendation(rec: dict, patient_info: str = ""):
    """Pretty print a single patient recommendation."""
    
    print("=" * 60)
    if patient_info:
        print(f"PATIENT: {patient_info}")
    print("=" * 60)
    
    print(f"\n{rec['summary']}")
    
    print(f"\n📊 ADMISSION RISK: {rec['admission']['risk_level']}")
    print(f"   Probability: {rec['admission']['probability']:.1%}")
    print("   Actions:")
    for action in rec['admission']['actions']:
        print(f"     {action}")
    
    print(f"\n⚠️  LWBS RISK: {rec['lwbs']['risk_tier']}")
    print(f"   Probability: {rec['lwbs']['probability']:.1%}")
    print(f"   Check at: {rec['lwbs']['check_time_min']} minutes")
    print("   Actions:")
    for action in rec['lwbs']['actions']:
        print(f"     {action}")
    
    print(f"\n🎯 PRIORITY SCORE: {rec['priority_score']:.2f}")
    print(f"   Overall Urgency: {rec['urgency']}")


def print_protocol_reference():
    """Print quick reference guide for Triage Lead."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TRIAGE LEAD QUICK REFERENCE                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🔴 ADMISSION RISK                                                           ║
║  ─────────────────                                                           ║
║  HIGH (≥68%)   → Start bed search. Notify admitting. Early consult.          ║
║  MEDIUM (50-67%) → Monitor closely. Prepare paperwork.                       ║
║  LOW (<50%)    → Standard pathway.                                           ║
║                                                                              ║
║  ⚠️  LWBS RISK (Watch List)                                                  ║
║  ──────────────────────────                                                  ║
║  TOP 5%  → Check at 15 min. Proactive communication.                         ║
║  TOP 10% → Check at 25 min. Update wait time.                                ║
║  TOP 15% → Check at 30 min.                                                  ║
║                                                                              ║
║  ⏱️  TIMING                                                                  ║
║  ──────────                                                                  ║
║  At triage    → Flag HIGH admission, identify TOP 10% LWBS                   ║
║  +15 min      → Check TOP 5% LWBS patients                                   ║
║  +25 min      → Check TOP 10% LWBS, re-assess MEDIUM admission               ║
║  +30 min      → Update all flagged patients                                  ║
║                                                                              ║
║  ❌ DO NOT                                                                   ║
║  ─────────                                                                   ║
║  • Trust every HIGH flag (55% are false alarms)                              ║
║  • Use LWBS as binary yes/no                                                 ║
║  • Promise exact wait times                                                  ║
║  • Alert on every prediction (causes fatigue)                                ║
║                                                                              ║
║  📊 EXPECTED PER 100 PATIENTS                                                ║
║  ────────────────────────────                                                ║
║  • ~14 will be admitted, model catches ~9                                    ║
║  • ~1.5 will LWBS, model flags ~0.8 in TOP 10%                               ║
║  • ~5 false admission alerts to review                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    
    # Initialize protocol
    protocol = TriageProtocol()
    
    # Print reference guide
    print_protocol_reference()
    
    # Example patients
    print("\n" + "=" * 60)
    print("EXAMPLE PATIENT RECOMMENDATIONS")
    print("=" * 60)
    
    # Patient 1: High admission risk
    rec1 = protocol.get_recommendation(admission_prob=0.75, lwbs_prob=0.01)
    print_recommendation(rec1, "78M, CTAS 2, Ambulance, Yellow Zone")
    
    print("\n")
    
    # Patient 2: High LWBS risk
    rec2 = protocol.get_recommendation(admission_prob=0.15, lwbs_prob=0.04)
    print_recommendation(rec2, "25F, CTAS 4, Walk-in, Green Zone")
    
    print("\n")
    
    # Patient 3: Both risks
    rec3 = protocol.get_recommendation(admission_prob=0.70, lwbs_prob=0.03)
    print_recommendation(rec3, "45M, CTAS 3, Ambulance, Yellow Zone")