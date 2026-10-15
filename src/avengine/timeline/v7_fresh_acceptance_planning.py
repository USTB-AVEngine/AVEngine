"""Plan untouched captures from legal inventories, without scene recipe inputs."""
from collections import defaultdict
from avengine.timeline.v7_family_constructors import stable_seed


def spread_legal_rooms(contract,houses,targets,seed):
    """One physical house per row; highest-area legal room first within a house."""
    from shapely.geometry import shape
    catalog={h['house']:h for h in houses};pool=defaultdict(lambda:defaultdict(list))
    for room in contract['rooms']:
        house=catalog.get(room['house'])
        if not house or house.get('split')!='train'or room['house']in contract.get('test_houses',[]):continue
        if 'standing_human'not in room['categories']:continue
        if room['room_id']not in house['checked_room_ids']:continue
        pool[room['family']][room['house']].append(room)
    chosen=[]
    for family,count in targets.items():
        ids=sorted(pool[family],key=lambda key:stable_seed(seed,family,key))
        if len(ids)<count:raise ValueError('legal independent house inventory below requested '+family+' count')
        for name in ids[:count]:
            def size(room):
                native=next(r for r in catalog[name]['checked_rooms']if r.get('room_label',r.get('room_id'))==room['room_id'])
                polygon=native.get('floor_polygon',native.get('room_floor_polygon'))
                return shape(polygon).area if polygon else 0.
            rooms=sorted(pool[family][name],key=lambda r:(-size(r),r['room_id']))
            chosen.append({'family':family,'house':name,'room':rooms[0],'alternative_room_ids':[r['room_id']for r in rooms[1:]],
                'scene_seed':stable_seed(seed,family,name),'handwritten_scene_configuration_inputs_read':[]})
    # Interleave datasets for the first smoke and early batch results.
    return sorted(chosen,key=lambda r:stable_seed(seed,'launch',r['house']))
