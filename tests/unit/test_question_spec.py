"""QuestionSpec 与 SceneRequirement 单元测试（任务三）。"""

from __future__ import annotations

import pytest

from avengine.qa.question_spec import (
    QuestionSpec,
    SceneRequirement,
    TemplateVariable,
    extract_scene_requirement,
    instantiate_template,
    list_template_variables,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════


def _make_spec(**overrides) -> QuestionSpec:
    """创建带默认值的 QuestionSpec。"""
    defaults = {
        "spec_id": "qs_001",
        "question_type": "sound_presence",
        "template": "穿{top_color}上衣的人是否在{time_window}发声？",
        "answer_modality": "sound_facts",
        "required_actor_count": 1,
    }
    defaults.update(overrides)
    return QuestionSpec(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# QuestionSpec 构造
# ═══════════════════════════════════════════════════════════════════════════════


class TestQuestionSpecConstruction:
    """QuestionSpec 构造与校验。"""

    def test_minimal_valid_spec(self):
        """最简合法构造。"""
        spec = QuestionSpec(
            spec_id="qs_001",
            question_type="sound_presence",
            template="是否发声？",
        )
        assert spec.spec_id == "qs_001"
        assert spec.question_type == "sound_presence"
        assert spec.answer_modality == "sound_facts"
        assert spec.required_actor_count == 1
        assert spec.time_window is None

    def test_with_time_window(self):
        """带时间窗口的构造。"""
        spec = QuestionSpec(
            spec_id="qs_tw",
            question_type="sound_presence",
            template="是否发声？",
            time_window=(0, 96000),
        )
        assert spec.time_window == (0, 96000)

    def test_rejects_empty_spec_id(self):
        """spec_id 为空应抛出异常。"""
        with pytest.raises(ValueError, match="spec_id"):
            QuestionSpec(spec_id="", question_type="t", template="t")

    def test_rejects_empty_question_type(self):
        """question_type 为空应抛出异常。"""
        with pytest.raises(ValueError, match="question_type"):
            QuestionSpec(spec_id="q1", question_type="", template="t")

    def test_rejects_empty_template(self):
        """template 为空应抛出异常。"""
        with pytest.raises(ValueError, match="template"):
            QuestionSpec(spec_id="q1", question_type="t", template="")

    def test_rejects_invalid_actor_count(self):
        """required_actor_count 必须 >= 1。"""
        with pytest.raises(ValueError, match="required_actor_count"):
            QuestionSpec(spec_id="q1", question_type="t", template="t",
                         required_actor_count=0)

    def test_rejects_invalid_modality(self):
        """非法 answer_modality 应抛出异常。"""
        with pytest.raises(ValueError, match="answer_modality"):
            QuestionSpec(spec_id="q1", question_type="t", template="t",
                         answer_modality="invalid")

    def test_rejects_invalid_time_window(self):
        """时间窗口 start > end 应抛出异常。"""
        with pytest.raises(ValueError, match="time_window"):
            QuestionSpec(spec_id="q1", question_type="t", template="t",
                         time_window=(100, 50))

    def test_negative_time_window_rejected(self):
        """时间窗口 start < 0 应抛出异常。"""
        with pytest.raises(ValueError, match="time_window"):
            QuestionSpec(spec_id="q1", question_type="t", template="t",
                         time_window=(-1, 50))

    def test_frozen_dataclass(self):
        """QuestionSpec 不可变。"""
        spec = _make_spec()
        with pytest.raises(Exception):
            spec.spec_id = "changed"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════════
# 模板变量
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateVariables:
    """模板变量提取与推断。"""

    def test_extracts_variable_names(self):
        spec = _make_spec(template="穿{top_color}上衣的{species_id}")
        assert spec.variable_names == ["top_color", "species_id"]

    def test_no_variables(self):
        spec = _make_spec(template="这是固定文本无变量")
        assert spec.variable_names == []

    def test_list_template_variables_basic(self):
        spec = _make_spec(template="穿{top_color}上衣的人")
        vars_ = list_template_variables(spec)
        assert len(vars_) == 1
        assert vars_[0].name == "top_color"
        assert vars_[0].source == "actor_attr"

    def test_time_window_variable(self):
        spec = _make_spec(template="{time_window}内是否发声？")
        vars_ = list_template_variables(spec)
        assert len(vars_) == 1
        assert vars_[0].name == "time_window"
        assert vars_[0].source == "time"

    def test_duplicate_variables_deduplicated(self):
        spec = _make_spec(template="{top_color}和{top_color}")
        vars_ = list_template_variables(spec)
        assert len(vars_) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 模板实例化
# ═══════════════════════════════════════════════════════════════════════════════


class TestInstantiateTemplate:
    """模板实例化。"""

    def test_single_variable(self):
        spec = _make_spec(template="穿{top_color}上衣")
        result = instantiate_template(spec, {"top_color": "蓝色"})
        assert result == "穿蓝色上衣"

    def test_multiple_variables(self):
        spec = _make_spec(template="{species_id}在{time_window}发声")
        result = instantiate_template(
            spec,
            {"species_id": "狗", "time_window": "0-3秒"},
        )
        assert result == "狗在0-3秒发声"

    def test_missing_binding_raises(self):
        spec = _make_spec(template="穿{top_color}上衣")
        with pytest.raises(ValueError, match="top_color"):
            instantiate_template(spec, {})

    def test_extra_bindings_ignored(self):
        """多余的绑定值被忽略，不报错。"""
        spec = _make_spec(template="穿{top_color}上衣")
        result = instantiate_template(
            spec,
            {"top_color": "蓝色", "extra": "忽略"},
        )
        assert result == "穿蓝色上衣"


# ═══════════════════════════════════════════════════════════════════════════════
# SceneRequirement 提取
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractSceneRequirement:
    """SceneRequirement 提取。"""

    def test_basic_extraction(self):
        spec = _make_spec()
        req = extract_scene_requirement(spec, {"top_color": "blue"})
        assert req.spec_id == "qs_001"
        assert req.target_attributes == {"top_color": "blue"}
        assert req.min_actors_in_scene == 1
        assert req.attribute_uniqueness_required is True

    def test_unknown_attribute_rejected(self):
        """未知属性名应抛出异常。"""
        spec = _make_spec()
        with pytest.raises(ValueError, match="unknown_attr"):
            extract_scene_requirement(spec, {"unknown_attr": "x"})

    def test_custom_time_window(self):
        spec = _make_spec()
        req = extract_scene_requirement(
            spec, {"top_color": "blue"},
            time_window=(0, 96000),
        )
        assert req.time_window == (0, 96000)

    def test_inherits_spec_time_window(self):
        spec = _make_spec(time_window=(48000, 96000))
        req = extract_scene_requirement(spec, {"top_color": "blue"})
        assert req.time_window == (48000, 96000)

    def test_override_over_spec_time_window(self):
        spec = _make_spec(time_window=(48000, 96000))
        req = extract_scene_requirement(
            spec, {"top_color": "blue"},
            time_window=(0, 48000),
        )
        assert req.time_window == (0, 48000)

    def test_custom_min_actors(self):
        spec = _make_spec(required_actor_count=3)
        req = extract_scene_requirement(spec, {"species_id": "human"})
        assert req.min_actors_in_scene == 3

    def test_can_disable_uniqueness(self):
        spec = _make_spec()
        req = extract_scene_requirement(
            spec, {"top_color": "blue"},
            attribute_uniqueness_required=False,
        )
        assert req.attribute_uniqueness_required is False


# ═══════════════════════════════════════════════════════════════════════════════
# SceneRequirement 不可变性
# ═══════════════════════════════════════════════════════════════════════════════


class TestSceneRequirement:
    """SceneRequirement 数据类型。"""

    def test_default_values(self):
        req = SceneRequirement(spec_id="qs1")
        assert req.target_attributes == {}
        assert req.required_sound_type is None
        assert req.time_window is None
        assert req.min_actors_in_scene == 1
        assert req.attribute_uniqueness_required is True

    def test_frozen(self):
        req = SceneRequirement(spec_id="qs1")
        with pytest.raises(Exception):
            req.spec_id = "changed"  # type: ignore[misc]
