"""Build free navigation from current UE collision geometry and null areas.

The companion C++ utility uses an explicitly installed Recast/Detour prefix.
It writes Habitat's existing MSET v2 format and retains the UE export reference.
Independent native loading, route and collision checks decide acceptance.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ue-export',required=True,type=Path)
    parser.add_argument('--recast-prefix',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--cxx',default='/usr/bin/g++')
    args=parser.parse_args()
    receipt_path=args.ue_export.resolve()
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('status')!='exported_inputs':
        raise ValueError('UE export did not produce navigation inputs')
    if not str(receipt.get('engine_version','')).startswith('5.5.'):
        raise ValueError('The detail-sampling defaults must be checked for this UE version')
    exports=receipt.get('navigation_geometry_exports',[])
    actors=receipt.get('navigation_actors',[])
    if len(exports)!=1 or len(actors)!=1:
        raise ValueError('Select exactly one exported navigation agent')
    source=Path(exports[0]).resolve()
    if not source.is_file():raise FileNotFoundError(source)
    error=float(actors[0]['properties']['max_simplification_error'])
    prefix=args.recast_prefix.resolve()
    include=prefix/'include/recastnavigation'
    libraries=[prefix/'lib/libRecast.a',prefix/'lib/libDetour.a']
    for path in [include/'Recast.h',include/'DetourNavMesh.h',*libraries]:
        if not path.is_file():raise FileNotFoundError(path)
    output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    binary=output/'compile_ue_navigation'
    cpp=Path(__file__).with_suffix('.cpp').resolve()
    compile_command=[args.cxx,'-O2','-std=c++17','-I'+str(include),str(cpp),*[str(p) for p in libraries],'-o',str(binary)]
    with (output/'build.log').open('x') as log:
        subprocess.run(compile_command,stdout=log,stderr=subprocess.STDOUT,check=True)
    navmesh=output/'navigation.navmesh'
    command=[str(binary),str(source),str(navmesh),str(error)]
    result=subprocess.run(command,text=True,capture_output=True)
    (output/'navigation_build.log').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr)
    measurement=json.loads(result.stdout)
    record={'status':'built_unverified','ue_export':str(receipt_path),'source_map':receipt['source_map'],
            'navigation_collision_geometry':str(source),'installed_recast_prefix':str(prefix),
            'navigation_mesh':str(navmesh),'builder_source':str(cpp),
            'commands':{'compile':compile_command,'build':command},'build':measurement,
            'coordinates':'UE Recast (-X,Z,-Y) centimetres to shared (X,Z,Y) metres.',
            'null_areas':'Exported convex null areas are applied after erosion. UE already expands their bounds; no second expansion is applied.',
            'detail_sampling_provenance':'UE 5.5 RecastNavMeshGenerator uses 600 cm sampling distance and 1 cm maximum error.',
            'claim_boundary':'A reconstructed Recast mesh from the current UE navigation inputs. It is not a byte copy of UE Detour tiles and is not accepted until independent native loading, route and collision checks pass.'}
    path=output/'navigation_build.json';path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'receipt':str(path),'navigation_mesh':str(navmesh)}))


if __name__=='__main__':main()
