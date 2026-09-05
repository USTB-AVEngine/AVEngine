# Room authoring interface

`build_room.py` is the reusable Blender entry point. It accepts a RoomSpec and a fresh output root:

```text
python3 tools/rooms/authoring/build_room.py \
  --spec examples/rooms/aea_loc3_social_rebuild/room_spec.json \
  --output-root /data/avengine_external/workspaces/multi_home_activity_20260905/room_a
```

The builder emits an editable `.blend`, visual and collision GLBs, preview renders, and metadata. The preview camera is authoring-only; production camera/listener planning remains in AVEngine. The copied source project is provenance only.
