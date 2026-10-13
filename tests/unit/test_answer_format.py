import unittest
from avengine.qa.answer_format import (ANGLE_INSTRUCTION, LETTER_INSTRUCTION, NUMBER_INSTRUCTION,
    PAIR_INSTRUCTION, DIRECTIONS, option_id, option_instruction, answer_text,
    format_prompt, normalize_export)
from avengine.qa.unified_scoring import score_mcq, score_counts


class AnswerFormatTests(unittest.TestCase):
    def test_option_boundary_and_every_index(self):
        for n in (2, 8, 21, 26, 27, 34):
            for i in range(n):
                label = option_id(i, n)
                self.assertRegex(label, r'^[A-Z]$' if n <= 26 else r'^[1-9]\d*$')
            self.assertEqual(option_instruction(n), LETTER_INSTRUCTION if n <= 26 else NUMBER_INSTRUCTION)
        self.assertEqual(option_id(25, 26), 'Z')
        self.assertEqual(option_id(26, 27), '27')

    def test_integer_angles_are_text_without_rounding(self):
        for n in (-180, -29, 0, 164, 179):
            self.assertEqual(answer_text(n, answer_type='angle_deg'), str(n))
        for invalid in (180, -181, 1.25, True):
            with self.assertRaises(ValueError):
                answer_text(invalid, answer_type='angle_deg')

    def test_every_choice_label_scores_correctly_with_existing_scorer(self):
        for n in (2, 8, 21, 26, 27, 34):
            options = [dict(value=f'class_{i}', label_en=f'class_{i}', label_zh=f'class_{i}') for i in range(n)]
            for i in range(n):
                form = dict(options=options, gold=dict(correct_index=i, value=f'class_{i}'))
                result = score_mcq(form, option_id(i, n))
                self.assertEqual(result['score'], 1.0, (n, i, result))

    def test_two_counts_keep_original_array_and_order(self):
        value = [2, 1]
        self.assertEqual(answer_text(value, answer_type='count_pair'), '2 1')
        self.assertEqual(value, [2, 1])
        plain, brackets = score_counts('2 1', value), score_counts('[2, 1]', value)
        self.assertEqual(plain['parsed'], brackets['parsed'])
        self.assertEqual(plain['score'], brackets['score'])
        self.assertEqual(plain['score'], 1.0)
        self.assertEqual(answer_text([0, 0], qa_id='QA-22'), '0 0')
        for invalid in ([2], [2, 1, 0], [2, -1]):
            with self.assertRaises(ValueError):
                answer_text(invalid, answer_type='count_pair')

    def test_count_and_direction_text(self):
        self.assertEqual(answer_text([2], answer_type='count_single'), '2')
        for text in DIRECTIONS:
            self.assertEqual(answer_text(text, qa_id='QA-04'), text)
        self.assertEqual(answer_text('front left', qa_id='QA-04'), 'front-left')
        self.assertEqual(answer_text('behind', qa_id='QA-28'), 'back')

    def test_prompt_and_export_agree_and_do_not_mutate_gold(self):
        source_public = [{'question_id': 'q', 'qa_id': 'QA-22', 'form': 'open',
                          'answer_type': 'count_pair', 'question_en': 'How many?'}]
        source_private = [{'question_id': 'q', 'canonical_answer': [2, 1],
                           'answer': [2, 1], 'evidence': {'unchanged': 1}}]
        public, private = normalize_export(source_public, source_private)
        self.assertEqual(private[0]['canonical_answer'], [2, 1])
        self.assertEqual(private[0]['answer'], [2, 1])
        self.assertEqual(private[0]['teacher_answer'], '2 1')
        self.assertEqual(source_private[0]['answer'], [2, 1])
        self.assertIn(PAIR_INSTRUCTION, public[0]['question_en'])
        angle = format_prompt('Where? Answer in whole degrees.', answer_type='angle_deg')
        self.assertIn(ANGLE_INSTRUCTION, angle)
        letter = format_prompt(angle, form='mcq', answer_type='angle_deg', option_count=26)
        self.assertIn(LETTER_INSTRUCTION, letter)
        self.assertNotIn(ANGLE_INSTRUCTION, letter)
        number = format_prompt('Which?', form='mcq', option_count=34)
        self.assertIn(NUMBER_INSTRUCTION, number)

    def test_reformatting_is_idempotent(self):
        p = [{'question_id':'q', 'qa_id':'QA-04', 'form':'mcq', 'question_en':'Which?',
              'choices':[{'id':'1','text':'front'}, {'id':'2','text':'back'}]}]
        g = [{'question_id':'q', 'canonical_answer':'back','answer':'2'}]
        once = normalize_export(p, g)
        self.assertEqual(normalize_export(*once), once)
        self.assertEqual(once[1][0]['answer'], 'B')

    def test_angle_export_keeps_the_original_authority_record(self):
        from avengine.qa.unified_catalog import _question_item
        from avengine.qa.v7_families import export_items
        item = _question_item(qa_id='QA-25', facts={'episode_id':'unit-angle'},
            seed='format-test', question_en='Which angle?', question_zh='角度？',
            open_answer_type='angle_deg', open_truth=163.9, truth_label='unused',
            evidence={'query_frame':15}, mcq_optional=True, slug='angle')
        self.assertEqual(item['truth']['value'], 164)
        self.assertEqual(item['truth']['label'], '164°')
        self.assertEqual(item['truth']['evidence']['answer_full_precision'], 163.9)
        self.assertEqual(item['forms']['open']['teacher_answer'], '164')
        public, private = export_items([item], 'unit-window', {})
        self.assertEqual(private[0]['authority_truth'], item['truth'])
        self.assertEqual(private[0]['canonical_answer'], 164)
        self.assertEqual(private[0]['answer'], 164)
        self.assertEqual(private[0]['teacher_answer'], '164')
        self.assertIn(ANGLE_INSTRUCTION, public[0]['question_en'])

    def test_exported_mcq_boundary_scores_with_original_scorer(self):
        from avengine.qa.unified_catalog import _question_item
        from avengine.qa.v7_families import export_items
        for n in (26, 27, 34):
            item = _question_item(qa_id='QA-21', facts={'episode_id':'unit-choices'},
                seed='format-test', question_en='Which class?', question_zh='类别？',
                open_answer_type='closed_set', open_truth='class_last', truth_label='class_last',
                evidence={}, slug='classes', options=[dict(value='class_last' if i==n-1 else f'class_{i}',
                    label_en='class_last' if i==n-1 else f'class_{i}') for i in range(n)])
            public, private = export_items([item], 'unit-window', {})
            q = next(v for v in public if v['form']=='mcq')
            a = next(v for v in private if v['question_id']==q['question_id'])
            self.assertEqual([c['id'] for c in q['choices']], [option_id(i,n) for i in range(n)])
            self.assertIn(option_instruction(n), q['question_en'])
            self.assertEqual(score_mcq(item['forms']['mcq'], a['answer'])['score'], 1.0)

if __name__ == '__main__':
    unittest.main()
