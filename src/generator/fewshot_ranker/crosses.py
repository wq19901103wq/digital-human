"""Versioned categorical interactions computed only from independently cached views."""
import json

from ..history_sources import require
from .features import equal

VERSION = 'local_crosses_v2'

# Deliberately selected low-cardinality interactions, not every pair of fields.
# In particular, do not cross raw person/chat identifiers with response content.
REPLY_CROSSES = {
    'needs_direct_answer': ('answers_own_context_request', 'directness', 'first_bubble_action'),
    'needs_clarification': ('secondary_action_clarify', 'secondary_action_ask'),
    'needs_acknowledgment': ('opening_style', 'first_bubble_action'),
    'needs_empathy': ('acknowledges_own_emotion', 'secondary_action_comfort', 'reply_tone'),
    'needs_coordination': ('advances_own_coordination', 'secondary_action_coordinate'),
    'needs_fact_boundary': ('unsupported_fact_signal', 'depends_on_unstated_fact'),
    'information_sufficiency': ('clarifies_own_ambiguity', 'explanation_level', 'depends_on_unstated_fact'),
    'referent_ambiguity': ('clarifies_own_ambiguity', 'secondary_action_ask'),
    'requires_external_fact': ('unsupported_fact_signal', 'depends_on_unstated_fact'),
    'has_unresolved_question': ('answers_own_context_request', 'question_bubble_position'),
    'has_pending_commitment': ('advances_own_coordination', 'closes_or_continues'),
    'has_correction_or_disagreement': ('reply_action', 'own_stance_continuity'),
    'relationship': ('reply_tone', 'humor_style', 'has_address', 'address_type'),
    'address_register': ('formality', 'address_position'),
    'chat_type': ('response_structure', 'bubble_count_bin', 'address_type'),
    'scene': ('reply_action', 'reply_tone', 'explanation_level'),
    'environment': ('bubble_count_bin', 'total_length_bin'),
    'urgency': ('directness', 'first_bubble_action', 'explanation_level'),
    'dialogue_stage': ('opening_style', 'closes_or_continues', 'response_structure'),
    'topic_shift': ('opening_style', 'own_timeline_continuity'),
    'emotion_valence': ('reply_tone', 'humor_style', 'acknowledges_own_emotion'),
    'formality': ('formality', 'humor_style'),
    'last_speaker_burst_length_bin': ('response_structure', 'first_bubble_action'),
    'latest_gap_bin': ('bubble_count_bin', 'opening_style'),
    'total_length_bin': ('total_length_bin', 'explanation_level'),
    'visible_participant_count_bin': ('reply_addressee_scope', 'has_address'),
    'has_question_mark': ('question_bubble_position', 'answers_own_context_request'),
}
CONTEXT_MATCHES = ('environment', 'address_register', 'addressee_scope', 'incoming_tone',
    'emotion_valence', 'urgency', 'information_sufficiency', 'referent_ambiguity',
    'requires_external_fact', 'has_unresolved_question', 'topic_shift')


def expand(row):
    """Accept the saved feature row, never answers, labels or raw target records."""
    result = dict(row)
    for target, reply_fields in REPLY_CROSSES.items():
        for reply in reply_fields:
            keys = ('target.' + target, 'example_reply.' + reply)
            require(all(isinstance(row.get(k), str) for k in keys), 'Missing categorical cross input')
            # JSON encoding is unambiguous even when a category contains delimiters.
            result[f'cross_v2.{target}_x_reply_{reply}'] = json.dumps(
                [row[k] for k in keys], ensure_ascii=False, separators=(',', ':'))
    for field in CONTEXT_MATCHES:
        keys = ('target.' + field, 'example_context.' + field)
        require(all(isinstance(row.get(k), str) for k in keys), 'Missing context match input')
        result[f'cross_v2.equal_context_{field}'] = equal(*(row[k] for k in keys))
    return result
