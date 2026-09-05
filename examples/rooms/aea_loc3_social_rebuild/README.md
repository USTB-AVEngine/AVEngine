# AEA loc3 social room rebuild

This is a research-candidate static room authoring input for the AEA loc3 dining/living reference. The reference informs layout scale and adjacency only; it does not provide the production camera. Production camera/listener planning remains in AVEngine.

The checked-in authoring entry is `tools/rooms/authoring/semantic_household_builder.py`. It accepts a room spec and writes an editable `.blend`, visual GLB, collision GLB, preview renders, object metadata, and seated-affordance metadata. The external Blender source project is provenance/reference input only.

Scope: static architecture, kitchen/living/dining furniture, four dining seats and sofa seating. Dynamic furniture and object manipulation are out of scope.

Fresh external output root:
`/data/avengine_external/workspaces/multi_home_activity_20260905/room_a`
