"""Declare breed-valid static animals and isolated Rocketbox coat variants."""
import argparse,json,copy
from pathlib import Path

COLORS={'black':'solid natural black','white':'solid pure white','brown':'solid rich chocolate brown','gray':'solid natural blue-gray, low saturation neutral gray','green':'natural green','blue':'natural sky blue','yellow':'natural clear yellow'}

def build_plan():
    rows=[]
    def body(kind,key,name,description,colors,size,source,measurement='shoulder_height_m',sound_group=None):
        for color in colors:
            aid=f'v7_{kind}_{key}_{color}_20261012'
            pose='a relaxed standing pose on four complete visibly separated legs and four paws on one level ground line; spine horizontal; one complete tail clear of the legs'
            if kind=='bird':pose='a natural upright perched pose, two complete clawed feet gripping one plain short horizontal wooden perch rod, both wings folded, full beak and complete tail visible; no vertical stem or base in the reference'
            prompt=(f'Studio product photograph of exactly one adult {name}. {description}. '
                    f'Coat or plumage: {COLORS[color]}. {pose}. '
                    'Three-quarter side view facing image-left, whole subject fully inside frame with generous blank margins. '
                    'Realistic anatomy and proportions, photorealistic fur or feather detail, matte natural surfaces, soft even neutral lighting, plain middle gray background. '
                    'No cage, collar, clothing, human, extra animal, extra limbs, extra head, metal, text, watermark, cropped body, body lying down, vertical quadruped, cartoon or sculpture.')
            rows.append({'asset_id':aid,'category':kind,'body_id':key,'body_name_zh':name,'body_description':description,'intended_color_label':color,'mode':'generated_static','walk_allowed':False,'prompt':prompt,'reference_seed':2026101200+len(rows),'mesh_seeds':[2026102200+len(rows)+i*1000 for i in range(3)],'physical_size':{'measurement':measurement,'target_m':size,'source_url':source,'status':'source_verification_pending'},'sound_appearance_family':sound_group})
    body('dog','standard_poodle','standard Poodle','Square athletic body, long slim muzzle and drop ears, short uniform curly pet clip showing all paws',['black','white','brown'],.52,'https://www.akc.org/dog-breeds/poodle-standard/')
    body('dog','labrador','Labrador Retriever','Broad head, strong medium-large body, short smooth coat, drop ears and thick otter tail',['black','brown'],.56,'https://www.akc.org/dog-breeds/labrador-retriever/')
    body('dog','newfoundland','Newfoundland dog','Large heavy broad body, broad head, thick dense medium-long coat and drop ears',['black','brown'],.68,'https://www.akc.org/dog-breeds/newfoundland/')
    body('dog','miniature_schnauzer','Miniature Schnauzer','Small square wiry body, natural folded ears, recognizable beard and eyebrows, short clipped torso and visible paws',['black','white'],.33,'https://ckcusa.com/breeds/miniature-schnauzer/')
    body('dog','havanese','Havanese dog','Small slightly elongated body, drop ears, silky wavy coat in a short puppy pet clip with legs visible, curved tail over back',['black','white','brown'],.25,'https://cdn.akc.org/Havanese.pdf')
    for key,name,desc,height in [
        ('british_shorthair','British Shorthair cat','Cobby compact sturdy body, broad round cheeks, small rounded ears, very dense plush short coat, thick medium tail',.26),
        ('exotic_shorthair','Exotic Shorthair cat','Stocky short-legged cobby body, broad flat rounded face, large round eyes, small round ears, dense short plush coat',.25),
        ('persian','Persian cat','Cobby broad body, broad flat round face, small ears, dense long fur in a neat natural coat with paws visible',.27),
        ('maine_coon','Maine Coon cat','Large long rectangular muscular body, square muzzle, tall tufted ears, shaggy long coat, long bushy tail',.32)]:
        body('cat',key,name,desc,['black','white','gray'],height,'https://www.fifeweb.org/breeds/standards/',sound_group='cat')
    body('bird','budgerigar','budgerigar parrot','Slender small parrot, small curved hooked beak, long tapering tail; green and blue forms have natural dark wing barring, yellow is lutino and white is albino',['green','blue','yellow','white'],.18,'https://lafeber.com/pet-birds/species/budgie/',measurement='beak_to_tail_length_m',sound_group='parrot')
    body('bird','fischer_lovebird','Fischer lovebird parrot','Compact stout small parrot, short rounded tail, prominent curved beak and white eye rings; green has orange face, blue has pale gray-white face, yellow is lutino with orange face and white is albino',['green','blue','yellow','white'],.15,'https://lovebirdgenetics.com/blog/fischer-lovebird-mutations/',measurement='beak_to_tail_length_m',sound_group='parrot')
    body('bird','border_canary','Border Fancy domestic canary','Small rounded full-bodied domestic canary, smooth uncrested round head, short straight conical beak, slender tail, two small pink feet',['yellow','white'],.14,'https://www.borderconvention.com/',measurement='beak_to_tail_length_m',sound_group='canary')
    body('bird','gloster_corona','Gloster Corona domestic canary','Very small compact stout domestic canary with a full rounded feather crown crest over the head, short straight conical beak and short neat tail',['yellow','white'],.115,'https://www.igba.co.uk/breeding-guide/',measurement='beak_to_tail_length_m',sound_group='canary')
    # Keep at least 16 labeled dogs; the two untagged Beagles are additional companions.
    for index,(body_id,color,variation) in enumerate([('standard_poodle','brown','a slightly narrower natural face and longer floppy ears'),('miniature_schnauzer','white','a fuller natural beard and compact angular muzzle')]):
        row=copy.deepcopy(next(x for x in rows if x['body_id']==body_id and x['intended_color_label']==color));row['asset_id']=f'v7_dog_{body_id}_{color}_variant2_20261012';row['reference_seed']=2026101350+index;row['mesh_seeds']=[2026102500+index+1000*i for i in range(3)];row['prompt']+=' This individual has '+variation+'.';row['individual_variant']=2;rows.append(row)
    for breed,coat,label in [('beagle','lemon_white',None),('beagle','red_white',None),('german_shepherd','solid_black','black'),('german_shepherd','solid_white','white')]:
        rows.append({'asset_id':f'v7_rocketbox_{breed}_{coat}_20261012','category':'dog','body_id':breed,'body_name_zh':breed,'coat_profile':coat,'intended_color_label':label,'mode':'rocketbox_coat_variant','walk_allowed':True,'source_roster_id':f'v7_rocketbox_{breed}_01','physical_size':{'measurement':'preserved_source_scale','source':'v7 roster'},'cross_requirement_exempt_companion':label is None})
    return {'schema':'avengine_v7_animal_variants_plan_v1','animal_count':len(rows),'animals':rows,'selected_label_colors':{'dog':['black','white','brown'],'cat':['black','white','gray'],'bird':['green','blue','yellow','white']},'minimum_individuals':{'dog':16,'cat':12,'bird':10},'minimum_bodies_per_label':{'dog':3,'cat':3,'bird':2},'minimum_labels_per_body':2,'max_seed_attempts':3,'reference_model':'black-forest-labs/FLUX.2-klein-4B','reference_revision':'e7b7dc27f91deacad38e78976d1f2b499d76a294','mesh_model':'TencentARC/Pixal3D','mesh_revision':'0b31f9160aa400719af409098bff7936a932f726','dino_revision':'3c276edd87d6f6e569ff0c4400e086807d0f3881'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);a=p.parse_args();plan=build_plan();(a.task_root/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');print({k:sum(x['category']==k for x in plan['animals']) for k in ['dog','cat','bird']})
if __name__=='__main__':main()
