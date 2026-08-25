# SPEAR 侧生成输入的镜像

**这是镜像，不是权威副本。** 工作副本在服务器
`/data/jzy/code/SPEAR-lead-b/data/controlled_source_attributes_v1/`，
路径结构与这里一一对应。

## 为什么要有这份镜像

`SPEAR-lead-b` 的 git 工作树是**断链**的：它的 gitdir 指向
`/data/jzy/code/AVEngine/external/SPEAR/.git/worktrees/SPEAR-lead-b`，
而那个父仓已经不存在，所以在那个检出里写的东西**一次都提交不了**。
profile 是生成输入——丢了就没法复现、也没法在同一个方法上继续做变体，
所以版本化的仓库里留一份。

## 里面是什么

- `candidate_profiles/static_object/audio_playback_*.json` —— 2026-08-26 为音响
  参考运行写的 5 份 `static_object` profile，都过了
  `controlled_source_asset_schema.validate_attribute_profile` 和 512 token 闸门。
- `candidate_profile_revisions/static_object/*/provenance.json` —— 由实测失败驱动的
  两份方法修订记录（书架箱单元数、电视可见出声口），格式照 SPEAR 仓里
  animal 那两份现成的写。

## 怎么用

把文件拷回 SPEAR 对应路径即可，两侧路径同构。改动请改 SPEAR 那份（工具从那里读），
然后重新跑一次镜像。**不要**让两边悄悄分叉。
