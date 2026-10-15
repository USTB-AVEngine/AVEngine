"""Rank original legal rooms by measured floor geometry, never certify a body."""


def floor_capacity(room,house):
    from shapely.geometry import shape
    actual=next((r for r in house.get('checked_rooms',[])if r.get('room_label',r.get('room_id'))==room['room_id']),{})
    polygon=actual.get('floor_polygon',actual.get('room_floor_polygon'))
    if not polygon:return 0.
    floor=shape(polygon)
    return float(floor.buffer(-.35).area)if floor.is_valid else 0.


def ranked_room_candidates(rooms,houses):
    return sorted(rooms,key=lambda pair:(-floor_capacity(pair[1],houses[pair[1]['house']]),pair[1]['house'],pair[1]['room_id']))


def required_categories(row,catalog,family):
    from avengine.timeline.v7_production_inputs import asset_category
    byid={a['asset_id']:a for a in catalog}
    return {asset_category(byid[aid],row.get('motion_requests',{}).get(aid,{}),family)for aid in row['asset_ids']}
