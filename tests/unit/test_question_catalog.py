"""测试 question_catalog.py 中十类问题 QuestionSpec 定义。

每个模板常量必须通过 QuestionSpec 构造校验，
且 ``answer_modality`` 在 _VALID_MODALITIES 集合内。
"""

from __future__ import annotations

import pytest

from avengine.qa.question_catalog import (
    ALL_QUESTION_SPECS,
    QS_ENTER_FRUSTUM,
    QS_OCCLUDER_IDENTITY,
    QS_SOUND_MOTION,
    QS_SOUND_ORDER,
    QS_SOUND_OVERLAP,
    QS_SOUND_PRESENCE,
    QS_SOUND_VISIBILITY,
    QS_SPATIAL_DIRECTION,
    QS_TRANSCRIPT,
    QS_TRANSCRIPT_TO_ATTR,
)
from avengine.qa.question_spec import (
    QuestionSpec,
    instantiate_template,
    list_template_variables,
)

ALL_SPECS = [
    QS_SOUND_PRESENCE,
    QS_TRANSCRIPT,
    QS_TRANSCRIPT_TO_ATTR,
    QS_SPATIAL_DIRECTION,
    QS_SOUND_ORDER,
    QS_SOUND_OVERLAP,
    QS_SOUND_MOTION,
    QS_ENTER_FRUSTUM,
    QS_SOUND_VISIBILITY,
    QS_OCCLUDER_IDENTITY,
]


class TestQuestionCatalogConstruction:
    """每个 QuestionSpec 常量构造正确。"""

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.spec_id)
    def test_is_question_spec(self, spec):
        assert isinstance(spec, QuestionSpec)

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.spec_id)
    def test_spec_id_non_empty(self, spec):
        assert spec.spec_id != ""

    @pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.spec_id)
    def test_template_has_variables(self, spec):
        """每类问题模板至少含一个变量。"""
        assert len(spec.variable_names) >= 1

    def test_all_question_specs_length(self):
        assert len(ALL_QUESTION_SPECS) == 10

    def test_all_question_specs_count(self):
        assert len(ALL_SPECS) == 10


class TestModalityMapping:
    """每类问题的 answer_modality 映射正确。"""

    def test_sound_presence(self):
        assert QS_SOUND_PRESENCE.answer_modality == "sound_facts"

    def test_transcript(self):
        assert QS_TRANSCRIPT.answer_modality == "sound_transcript"

    def test_transcript_to_attr(self):
        assert QS_TRANSCRIPT_TO_ATTR.answer_modality == "reverse_attribute"

    def test_spatial_direction(self):
        assert QS_SPATIAL_DIRECTION.answer_modality == "spatial_direction"

    def test_sound_order(self):
        assert QS_SOUND_ORDER.answer_modality == "sound_order"
        assert QS_SOUND_ORDER.required_actor_count == 2

    def test_sound_overlap(self):
        assert QS_SOUND_OVERLAP.answer_modality == "overlap_sound"
        assert QS_SOUND_OVERLAP.required_actor_count == 2

    def test_sound_motion(self):
        assert QS_SOUND_MOTION.answer_modality == "sound_motion"

    def test_enter_frustum(self):
        assert QS_ENTER_FRUSTUM.answer_modality == "enter_frustum_direction"

    def test_sound_visibility(self):
        assert QS_SOUND_VISIBILITY.answer_modality == "sound_visibility"

    def test_occluder_identity(self):
        assert QS_OCCLUDER_IDENTITY.answer_modality == "occluder_identity"


class TestTemplateVariables:
    """每类问题模板包含预期变量。"""

    def test_sound_presence_variables(self):
        assert "top_color" in QS_SOUND_PRESENCE.variable_names
        assert "species_id" in QS_SOUND_PRESENCE.variable_names
        assert "time_window" in QS_SOUND_PRESENCE.variable_names

    def test_transcript_variables(self):
        assert "top_color" in QS_TRANSCRIPT.variable_names

    def test_transcript_to_attr_variables(self):
        assert "transcript" in QS_TRANSCRIPT_TO_ATTR.variable_names

    def test_spatial_direction_variables(self):
        assert "top_color" in QS_SPATIAL_DIRECTION.variable_names

    def test_sound_order_variables(self):
        names = QS_SOUND_ORDER.variable_names
        assert "species_id_a" in names
        assert "species_id_b" in names

    def test_sound_overlap_variables(self):
        names = QS_SOUND_OVERLAP.variable_names
        assert "species_id_a" in names
        assert "species_id_b" in names

    def test_sound_motion_variables(self):
        assert "species_id" in QS_SOUND_MOTION.variable_names


class TestInstantiateTemplates:
    """模板实例化生成有意义的中文问题文本。"""

    def test_sound_presence_instantiate(self):
        text = instantiate_template(QS_SOUND_PRESENCE, {
            "top_color": "蓝色",
            "species_id": "人",
            "time_window": "0.0-2.0秒",
        })
        assert "蓝色" in text
        assert "人" in text
        assert "发声" in text

    def test_transcript_instantiate(self):
        text = instantiate_template(QS_TRANSCRIPT, {
            "top_color": "绿色",
            "species_id": "人",
            "time_window": "0.0-2.0秒",
        })
        assert "绿色" in text
        assert "说了什么" in text

    def test_transcript_to_attr_instantiate(self):
        text = instantiate_template(QS_TRANSCRIPT_TO_ATTR, {
            "transcript": "请关上门",
            "species_id": "人",
        })
        assert '"请关上门"' in text
        assert "穿什么颜色" in text
        assert "人" in text

    def test_spatial_direction_instantiate(self):
        text = instantiate_template(QS_SPATIAL_DIRECTION, {
            "top_color": "红色",
            "species_id": "人",
            "time_window": "0.0-2.0秒",
        })
        assert "方向" in text
        assert "红色" in text

    def test_sound_order_instantiate(self):
        text = instantiate_template(QS_SOUND_ORDER, {
            "species_id_a": "猫",
            "species_id_b": "狗",
            "time_window": "0.0-3.0秒",
        })
        assert "猫" in text
        assert "狗" in text
        assert "先发声" in text

    def test_sound_overlap_instantiate(self):
        text = instantiate_template(QS_SOUND_OVERLAP, {
            "species_id_a": "猫",
            "species_id_b": "狗",
        })
        assert "猫" in text
        assert "狗" in text
        assert "发声" in text

    def test_enter_frustum_instantiate(self):
        text = instantiate_template(QS_ENTER_FRUSTUM, {
            "species_id": "狗",
        })
        assert "狗" in text
        assert "进入" in text

    def test_sound_visibility_instantiate(self):
        text = instantiate_template(QS_SOUND_VISIBILITY, {
            "species_id": "暹罗猫",
            "time_window": "0.0-2.0秒",
        })
        assert "暹罗猫" in text
        assert "清晰可见" in text
        assert "遮挡" in text

    def test_occluder_identity_instantiate(self):
        text = instantiate_template(QS_OCCLUDER_IDENTITY, {
            "species_id": "狗",
        })
        assert "狗" in text
        assert "家具" in text
        assert "另一只动物" in text


class TestRequiredActorCount:
    """双角色问题的 required_actor_count == 2。"""

    def test_sound_presence_single_actor(self):
        assert QS_SOUND_PRESENCE.required_actor_count == 1

    def test_transcript_single_actor(self):
        assert QS_TRANSCRIPT.required_actor_count == 1

    def test_sound_order_dual_actor(self):
        assert QS_SOUND_ORDER.required_actor_count == 2

    def test_sound_overlap_dual_actor(self):
        assert QS_SOUND_OVERLAP.required_actor_count == 2
