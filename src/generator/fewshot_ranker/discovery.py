"""Hypothesis-driven feature families from the two frozen, independent views.

No label statistics enter a feature. Each family is ablated on the same split;
coverage diagnostics use fit only and do not decide from outer validation.
"""
from collections import Counter

from . import crosses
from .identity_crosses import pair
from .schema import UNKNOWN

VERSION = 'behavior_interactions_v1'
FAMILIES = ('response_need', 'continuity', 'rhythm', 'transfer')


def specifications():
    result = {name: [] for name in FAMILIES}

    def add(family, left, right):
        result[family].append((left, right))

    def replies(family, mapping):
        for target, fields in mapping.items():
            for reply in fields:
                if reply not in crosses.REPLY_CROSSES.get(target, ()):
                    add(family, 'target.'+target, 'example_reply.'+reply)

    replies('response_need', {
        'needs_direct_answer': ('last_bubble_action', 'explanation_level', 'secondary_action_answer'),
        'needs_clarification': ('first_bubble_action', 'question_bubble_position', 'directness'),
        'needs_acknowledgment': ('closes_or_continues', 'response_structure'),
        'needs_empathy': ('opening_style', 'emotion_intensity', 'directness'),
        'needs_coordination': ('response_structure', 'last_bubble_action'),
        'needs_fact_boundary': ('named_entity_density', 'secondary_action_ask'),
        'has_unresolved_question': ('last_bubble_action', 'secondary_action_answer'),
        'information_sufficiency': ('directness', 'secondary_action_ask'),
        'referent_ambiguity': ('named_entity_density', 'has_address'),
        **{f'pending_intent_{v}': ('reply_action', 'closes_or_continues')
           for v in ('answer', 'clarify', 'coordinate', 'comfort', 'acknowledge')},
    })
    replies('continuity', {
        'last_self_action': ('opening_style', 'last_bubble_action', 'repeats_context_needlessly'),
        'last_self_tone': ('humor_style', 'emotion_intensity'),
        'self_stance': ('reply_action', 'own_stance_continuity'),
        'has_pending_commitment': ('first_bubble_action', 'own_timeline_continuity'),
        'has_correction_or_disagreement': ('opening_style', 'directness'),
        'has_self_history': ('opening_style', 'repeats_context_needlessly'),
        'last_self_position_bin': ('opening_style', 'response_structure'),
        'incoming_tone': ('humor_style', 'directness', 'first_bubble_action'),
        'emotion_valence': ('opening_style', 'emotion_intensity'),
        'emotion_intensity': ('humor_style', 'directness'),
        'topic_shift': ('reuses_own_context_phrase', 'last_bubble_action'),
        'dialogue_stage': ('first_bubble_action', 'last_bubble_action'),
        'addressee_scope': ('reply_addressee_scope', 'has_address'),
    })
    replies('rhythm', {
        'speaker_switch_count_bin': ('bubble_count_bin', 'response_structure'),
        'last_speaker_run_length_bin': ('bubble_count_bin', 'question_bubble_position'),
        'last_speaker_burst_length_bin': ('first_bubble_length_bin', 'last_bubble_action'),
        'latest_gap_bin': ('first_bubble_length_bin', 'response_structure'),
        'last_self_position_bin': ('bubble_count_bin', 'total_length_bin'),
        'urgency': ('bubble_count_bin', 'first_bubble_length_bin'),
        'context_message_count_bin': ('bubble_count_bin', 'explanation_level'),
        'is_group_repetition_pattern': ('reuses_own_context_phrase', 'first_bubble_action'),
        'visible_participant_count_bin': ('address_position', 'bubble_count_bin'),
        'has_question_mark': ('response_structure', 'first_bubble_length_bin'),
        'has_laughter': ('has_laughter', 'humor_style'),
        'has_emoji': ('has_emoji', 'reply_tone'),
        'has_particle': ('has_particle', 'formality'),
    })
    for left, right in (('first_bubble_action', 'last_bubble_action'),
                        ('bubble_count_bin', 'response_structure'),
                        ('first_bubble_length_bin', 'last_bubble_length_bin')):
        add('rhythm', 'example_reply.'+left, 'example_reply.'+right)
    for source in ('history_age_bin', 'lexical_jaccard'):
        for match in ('equal_chat_id', 'equal_topic_domain', 'equal_primary_intent', 'equal_scene'):
            add('transfer', 'cross.'+source, 'cross.'+match)
        for field in ('named_entity_density', 'depends_on_unstated_fact', 'unsupported_fact_signal',
                      'own_context_fit', 'reuses_own_context_phrase'):
            add('transfer', 'cross.'+source, 'example_reply.'+field)
    replies('transfer', {
        'topic_domain': ('named_entity_density', 'depends_on_unstated_fact'),
        'requires_external_fact': ('named_entity_density', 'own_context_fit'),
        'information_sufficiency': ('own_context_fit', 'unsupported_fact_signal'),
    })
    for source in ('equal_topic_domain', 'equal_chat_id', 'needs_jaccard'):
        for field in ('depends_on_unstated_fact', 'own_context_fit'):
            add('transfer', 'cross.'+source, 'example_reply.'+field)
    return result


SPECS = specifications()


def fields(row, families=FAMILIES):
    """FG needs cached categories only; answers, labels and target text are absent."""
    return {f'discovery.{family}.{left}_x_{right}': pair(row, left, right)
            for family in families for left, right in SPECS[family]}


def profile(groups, observations, additions):
    """Identify sparse/unavailable fields and those that can distinguish candidates."""
    fit = [g for g in groups if g['split'] == 'fit']
    indices = [i for g in fit for i in g['indices']]
    mixed = [g for g in fit if {observations[i]['z'] for i in g['indices']} == {0, 1}]
    values = {}
    for key in additions[0]:
        counts = Counter(additions[i][key] for i in indices)
        known = sum(count for v, count in counts.items() if v != UNKNOWN)
        values[key] = dict(known_rows=known, total_rows=len(indices),
            known_fraction=known/len(indices), known_categories=len(set(counts)-{UNKNOWN}),
            categories_supported_by_at_least_10_rows=sum(v != UNKNOWN and n >= 10 for v, n in counts.items()),
            mixed_contexts_with_distinct_values=sum(
                len({additions[i][key] for i in g['indices']}) > 1 for g in mixed))
    return dict(scope='fit only; coverage does not select features', fit_contexts=len(fit),
                mixed_fit_contexts=len(mixed), fields=values,
                family_sizes={name: len(specs) for name, specs in SPECS.items()})
