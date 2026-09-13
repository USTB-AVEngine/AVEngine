// Build a Habitat-readable navigation mesh from Unreal's ExportNavigation input.
// The Recast construction sequence follows native/habitat/esp/nav/PathFinder.cpp
// (Meta Platforms, MIT), adding the exported null-area modifiers before regions.
// Recast and Detour are linked from an explicit installed dependency prefix.
#include <Recast.h>
#include <DetourNavMesh.h>
#include <DetourNavMeshBuilder.h>
#include <DetourAlloc.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

struct Area { int id; float low, high; std::vector<float> points; };
// Existing Habitat MSET v2 settings layout (13 floats followed by four bools).
// Its native loader is used to verify every field after writing.
struct Settings {
 float cellSize, cellHeight, agentHeight, agentRadius, agentMaxClimb, agentMaxSlope;
 float regionMinSize, regionMergeSize, edgeMaxLen, edgeMaxError, vertsPerPoly;
 float detailSampleDist, detailSampleMaxError;
 bool filterLowHangingObstacles, filterLedgeSpans, filterWalkableLowHeightSpans, includeStaticObjects;
};
static_assert(sizeof(Settings)==56, "Unexpected Habitat settings layout");
struct Header { int magic, version, tiles; dtNavMeshParams params; };
struct TileHeader { dtTileRef ref; int size; };
static void require(bool condition, const char* message) { if(!condition) throw std::runtime_error(message); }

int main(int argc, char** argv) try {
 require(argc==4, "Usage: compile_ue_navigation input.obj output.navmesh max_simplification_error");
 std::ifstream in(argv[1]); require(bool(in),"Cannot open navigation export");
 std::vector<float> verts; std::vector<int> tris; std::vector<Area> areas;
 std::map<std::string,float> fields; std::array<float,6> bounds{}; bool haveBounds=false;
 std::string line;
 while(std::getline(in,line)) {
  std::istringstream row(line); std::string key; row>>key;
  if(key=="v") {float x,y,z; require(bool(row>>x>>y>>z),"Invalid vertex"); verts.insert(verts.end(),{-x*.01f,y*.01f,-z*.01f});}
  else if(key=="f") {int a,b,c; require(bool(row>>a>>b>>c),"Invalid triangle"); tris.insert(tris.end(),{a-1,b-1,c-1});}
  else if(key=="AE") {
   Area a{};int count;require(bool(row>>a.id>>count>>a.low>>a.high),"Invalid area");
   require(a.id==0 && count>=3,"Only explicitly exported null areas are supported");
   a.low*=.01f;a.high*=.01f;
   for(int i=0;i<count;i++) {require(bool(std::getline(in,line)),"Truncated area");std::istringstream point(line);float x,y,z;require(bool(point>>key>>x>>y>>z)&&key=="Av","Invalid area point");a.points.insert(a.points.end(),{-x*.01f,y*.01f,-z*.01f});}
   areas.push_back(a);
  } else if(key=="rd_bbox") {
   float x0,y0,z0,x1,y1,z1;require(bool(row>>x0>>y0>>z0>>x1>>y1>>z1),"Invalid bounds");bounds={-x1*.01f,y0*.01f,-z1*.01f,-x0*.01f,y1*.01f,-z0*.01f};haveBounds=true;
  } else if(key.rfind("rd_",0)==0) {float value;require(bool(row>>value),"Invalid parameter");fields[key]=value;}
 }
 auto get=[&](const char* key){auto it=fields.find(key);require(it!=fields.end(),key);return it->second;};
 require(haveBounds && !verts.empty() && !tris.empty(),"Navigation geometry is empty");
 for(float x:verts)require(std::isfinite(x),"Non-finite vertex");
 for(int i:tris)require(i>=0 && i<int(verts.size()/3),"Invalid triangle index");
 rcConfig cfg{};cfg.cs=get("rd_cs")*.01f;cfg.ch=get("rd_ch")*.01f;
 require(cfg.cs>0 && cfg.ch>0,"Invalid voxel sizes");
 cfg.walkableSlopeAngle=get("rd_ams");cfg.walkableHeight=int(std::ceil(get("rd_agh")/get("rd_ch")));
 cfg.walkableClimb=int(std::floor(get("rd_amc")/get("rd_ch")));cfg.walkableRadius=int(std::ceil(get("rd_agr")/get("rd_cs")));
 cfg.maxEdgeLen=int(get("rd_mel"));cfg.maxSimplificationError=std::stof(argv[3]);
 cfg.minRegionArea=int(get("rd_rmis")*get("rd_rmis"));cfg.mergeRegionArea=int(get("rd_rmas")*get("rd_rmas"));cfg.maxVertsPerPoly=int(get("rd_mvpp"));
 // UE 5.5 generator constants are 600 cm sampling distance and 1 cm error.
 cfg.detailSampleDist=6.0f;cfg.detailSampleMaxError=.01f;
 rcVcopy(cfg.bmin,bounds.data());rcVcopy(cfg.bmax,bounds.data()+3);rcCalcGridSize(cfg.bmin,cfg.bmax,cfg.cs,&cfg.width,&cfg.height);
 require(cfg.width>0 && cfg.height>0 && int64_t(cfg.width)*cfg.height<100000000,"Invalid navigation grid dimensions");
 rcContext ctx;auto hf=rcAllocHeightfield();require(hf,"Cannot allocate heightfield");
 require(rcCreateHeightfield(&ctx,*hf,cfg.width,cfg.height,cfg.bmin,cfg.bmax,cfg.cs,cfg.ch),"Cannot create heightfield");
 std::vector<unsigned char> walk(tris.size()/3,0);rcMarkWalkableTriangles(&ctx,cfg.walkableSlopeAngle,verts.data(),int(verts.size()/3),tris.data(),int(tris.size()/3),walk.data());
 require(rcRasterizeTriangles(&ctx,verts.data(),int(verts.size()/3),tris.data(),walk.data(),int(tris.size()/3),*hf,cfg.walkableClimb),"Cannot rasterize navigation input");
 rcFilterLowHangingWalkableObstacles(&ctx,cfg.walkableClimb,*hf);
 rcFilterLedgeSpans(&ctx,cfg.walkableHeight,cfg.walkableClimb,*hf);
 rcFilterWalkableLowHeightSpans(&ctx,cfg.walkableHeight,*hf);
 auto chf=rcAllocCompactHeightfield();require(chf,"Cannot allocate compact heightfield");
 require(rcBuildCompactHeightfield(&ctx,cfg.walkableHeight,cfg.walkableClimb,*hf,*chf),"Cannot compact navigation input");rcFreeHeightField(hf);
 require(rcErodeWalkableArea(&ctx,cfg.walkableRadius,*chf),"Cannot erode navigation area");
 // ExportNavigation already expands these convex footprints by agent radius
 // and their vertical bounds by one cell; do not expand them a second time.
 for(const auto& a:areas)rcMarkConvexPolyArea(&ctx,a.points.data(),int(a.points.size()/3),a.low,a.high,RC_NULL_AREA,*chf);
 require(rcBuildDistanceField(&ctx,*chf),"Cannot build navigation distances");
 require(rcBuildRegions(&ctx,*chf,0,cfg.minRegionArea,cfg.mergeRegionArea),"Cannot build navigation regions");
 auto contours=rcAllocContourSet();require(contours,"Cannot allocate contours");
 require(rcBuildContours(&ctx,*chf,cfg.maxSimplificationError,cfg.maxEdgeLen,*contours),"Cannot build contours");
 auto poly=rcAllocPolyMesh();require(poly,"Cannot allocate polygons");require(rcBuildPolyMesh(&ctx,*contours,cfg.maxVertsPerPoly,*poly),"Cannot build polygons");
 auto detail=rcAllocPolyMeshDetail();require(detail,"Cannot allocate detail mesh");require(rcBuildPolyMeshDetail(&ctx,*poly,*chf,cfg.detailSampleDist,cfg.detailSampleMaxError,*detail),"Cannot build detail mesh");
 require(poly->npolys>0,"No walkable polygons");
 for(int i=0;i<poly->npolys;i++){require(poly->areas[i]==RC_WALKABLE_AREA,"Unexpected remaining area");poly->areas[i]=0;poly->flags[i]=1;}
 dtNavMeshCreateParams params{};params.verts=poly->verts;params.vertCount=poly->nverts;params.polys=poly->polys;params.polyAreas=poly->areas;params.polyFlags=poly->flags;params.polyCount=poly->npolys;params.nvp=poly->nvp;
 params.detailMeshes=detail->meshes;params.detailVerts=detail->verts;params.detailVertsCount=detail->nverts;params.detailTris=detail->tris;params.detailTriCount=detail->ntris;
 params.walkableHeight=get("rd_agh")*.01f;params.walkableRadius=get("rd_agr")*.01f;params.walkableClimb=get("rd_amc")*.01f;rcVcopy(params.bmin,poly->bmin);rcVcopy(params.bmax,poly->bmax);params.cs=cfg.cs;params.ch=cfg.ch;params.buildBvTree=true;
 unsigned char* data=nullptr;int size=0;require(dtCreateNavMeshData(&params,&data,&size),"Cannot serialize Detour data");
 auto mesh=dtAllocNavMesh();require(mesh,"Cannot allocate Detour mesh");require(dtStatusSucceed(mesh->init(data,size,DT_TILE_FREE_DATA)),"Cannot initialize Detour mesh");
 Header header{};header.magic='M'<<24|'S'<<16|'E'<<8|'T';header.version=2;header.tiles=1;header.params=*mesh->getParams();
 Settings settings{cfg.cs,cfg.ch,params.walkableHeight,params.walkableRadius,params.walkableClimb,cfg.walkableSlopeAngle,get("rd_rmis"),get("rd_rmas"),cfg.maxEdgeLen*cfg.cs,cfg.maxSimplificationError,float(cfg.maxVertsPerPoly),cfg.detailSampleDist/cfg.cs,cfg.detailSampleMaxError/cfg.ch,true,true,true,false};
 const dtMeshTile* tile=static_cast<const dtNavMesh*>(mesh)->getTile(0);TileHeader th{};th.ref=mesh->getTileRef(tile);th.size=tile->dataSize;
 int fd=open(argv[2],O_WRONLY|O_CREAT|O_EXCL,0664);require(fd>=0,"Output must be fresh and writable");FILE* fp=fdopen(fd,"wb");require(fp,"Cannot open output stream");
 require(fwrite(&header,sizeof(header),1,fp)==1 && fwrite(&settings,sizeof(settings),1,fp)==1 && fwrite(&th,sizeof(th),1,fp)==1 && fwrite(tile->data,tile->dataSize,1,fp)==1,"Cannot write navigation mesh");require(fclose(fp)==0,"Cannot close navigation mesh");
 std::cout<<"{\"status\":\"built_unverified\",\"vertices\":"<<verts.size()/3<<",\"triangles\":"<<tris.size()/3<<",\"null_areas\":"<<areas.size()<<",\"polygons\":"<<poly->npolys<<",\"voxel_width\":"<<cfg.width<<",\"voxel_height\":"<<cfg.height<<"}\n";
 dtFreeNavMesh(mesh);rcFreePolyMeshDetail(detail);rcFreePolyMesh(poly);rcFreeContourSet(contours);rcFreeCompactHeightfield(chf);return 0;
} catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
