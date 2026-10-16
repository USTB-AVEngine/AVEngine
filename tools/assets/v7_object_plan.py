"""Declare shape-by-colour object requests before generation; independent silhouettes share each colour."""
import argparse,json
from pathlib import Path
from v7_object_prepare import dump

# Each tuple supplies six cells: shape A in colours 1/2, B in 2/3, C in 3/1.
SPECS={
'alarm_clock':(['digital_cube','twin_bell_analog','retro_rectangular'],['black','white','red'],[.09,.14,.12]),
'blender':(['jug_blender','bullet_blender','personal_smoothie_blender'],['black','white','silver'],[.22,.40,.20]),
'microwave_oven':(['countertop','over_range','retro_rounded'],['black','white','silver'],[.40,.29,.49]),
'printer':(['desktop_inkjet','office_laser_mfp','compact_thermal'],['black','white','gray'],[.35,.30,.42]),
'bookshelf_speaker':(['compact_shelf_cabinet','rounded_studio_monitor','sealed_wood_cabinet'],['black','white','wood_brown'],[.22,.33,.21]),
'floorstanding_speaker':(['floor_standing_tower','slim_column','curved_cabinet'],['black','white','wood_brown'],[.30,.95,.26]),
'smart_speaker':(['fabric_wrapped_cylinder','low_dome','rounded_cube'],['black','white','gray'],[.13,.16,.13]),
'soundbar':(['wide_low_bar','angular_driver_bar','rounded_caps_bar'],['black','white','gray'],[.10,.08,.75]),
'television':(['flat_panel_central_pedestal','flat_panel_splayed_feet','retro_crt'],['black','white','silver'],[.20,.60,.90]),
'desk_telephone':(['rotary_round_base','rotary_square_base','bell_pushbutton'],['black','red','beige'],[.20,.16,.20]),
'smoke_detector':(['ceiling_disc','wall_square','compact_rectangular'],['white','gray','beige'],[.12,.075,.12]),
'wall_clock':(['round_analog','square_analog','railway_round_analog'],['black','white','red'],[.04,.29,.29]),
'mantel_clock':(['round_on_feet','arched_wood_case','rectangular_analog'],['black','white','wood_brown'],[.08,.20,.23]),
'carbon_monoxide_alarm':(['carbon_monoxide_alarm','co_wide_display','co_rounded_vertical'],['white','black','gray'],[.045,.11,.13]),
'security_keypad':(['security_keypad','security_touch_panel','security_hinged_cover'],['white','black','gray'],[.04,.15,.12]),
'kitchen_timer':(['kitchen_timer','timer_round_digital','timer_clip_on'],['white','black','gray'],[.04,.085,.085]),
'radio':(['portable_rectangular','rounded_retro','wood_tabletop'],['black','white','wood_brown'],[.10,.17,.26]),
 'tablet':None}
SPECS['tablet']=(['slim_tablet','rugged_tablet','convertible_tablet'],['black','white','silver'],[.015,.24,.16])
DESCRIPTIONS={
'digital_cube':'a compact cube digital alarm clock with an inset dark display and top snooze button',
'twin_bell_analog':'an upright twin-bell mechanical alarm clock with circular dial, two bells, and two stable small feet',
'retro_rectangular':'an upright retro rectangular analog alarm clock on a wide stable foot',
'jug_blender':'a kitchen blender with motor base and transparent tall jug, lid, handle and front knob',
'bullet_blender':'a bullet smoothie blender with short motor base and one inverted transparent cup',
'personal_smoothie_blender':'a narrow personal blender with cylindrical motor base and upright capped transparent bottle',
'countertop':'a rectangular countertop microwave with broad flat side panels, front dark door window, and narrow right control strip',
'over_range':'a wide over-range microwave with rectangular wall mounting back, broad front dark door and right control strip',
'retro_rounded':'a retro rounded-corner countertop microwave with two front knobs, horizontal handle, dark door window and stable feet',
'desktop_inkjet':'a low rectangular desktop inkjet printer with front paper outlet and top paper input',
'office_laser_mfp':'an office laser multifunction printer with scanner lid and front paper drawer',
'compact_thermal':'a compact desktop thermal label printer with domed rectangular body, flat base and a front label outlet',
'compact_shelf_cabinet':'one compact upright bookshelf loudspeaker with rectangular cabinet, front tweeter and woofer',
'rounded_studio_monitor':'one upright rounded-corner studio monitor speaker, with a visible front woofer and tweeter',
'sealed_wood_cabinet':'one sealed upright rectangular bookshelf loudspeaker, with a broad wood cabinet and one round front driver',
'floor_standing_tower':'one tall floor-standing rectangular loudspeaker tower with three round front drivers and broad stable bottom',
'slim_column':'one slim floor-standing loudspeaker column with narrow drivers and a stable integrated plinth',
'curved_cabinet':'one floor-standing loudspeaker with curved side panels, two front drivers and stable pedestal plinth',
'fabric_wrapped_cylinder':'one smart speaker in a short upright fabric-wrapped cylinder, with flat round base and top controls',
'low_dome':'one low dome smart speaker with a flat base and fabric exterior',
'rounded_cube':'one smart speaker in a rounded cube body, fabric sides, flat base and small top buttons',
'wide_low_bar':'one horizontal television soundbar with a long straight rectangular body, flat base and dark front grille',
'angular_driver_bar':'one horizontal television soundbar with angled end caps, a flat base and two separate dark front driver panels',
'rounded_caps_bar':'one long horizontal soundbar with rounded end caps, flat base and a narrow front grille',
'flat_panel_central_pedestal':'one upright flatscreen television with a wide coloured cabinet bezel, dark blank screen, and broad central pedestal',
'flat_panel_splayed_feet':'one upright flatscreen television with a wide coloured cabinet bezel, dark blank screen, and two outward feet',
'retro_crt':'one retro CRT television with deep coloured cabinet, curved dark blank screen, small front speaker and flat base',
'rotary_round_base':'one old mechanical bell rotary-dial telephone with round base, circular finger dial, curved handset resting correctly on two cradle posts',
'rotary_square_base':'one old mechanical bell rotary-dial telephone with a square sloped base, circular finger dial and handset in its cradle',
'bell_pushbutton':'one vintage bell-ringing pushbutton landline desk telephone, rectangular sloped base and handset resting in its cradle',
'ceiling_disc':'one round ceiling-mounted smoke alarm disc, shallow body, a wide flat top mounting back and vents facing downward',
'wall_square':'one wall-mounted square smoke detector with flat mounting back, vented front and a small button',
'compact_rectangular':'one compact rectangular wall-mounted smoke detector with flat rear mounting plate and vented front',
'round_analog':'one wall-mounted analog clock with broad round coloured frame, blank white dial, clearly visible dark hour and minute hands and flat back',
'square_analog':'one wall-mounted analog clock with broad square coloured frame, white dial, clearly visible dark clock hands and flat back',
'railway_round_analog':'one railway-style wall-mounted analog clock with thick round coloured frame, visible dark clock hands, white dial and flat back',
'round_on_feet':'one upright tabletop analog mantel clock with round broad coloured case, visible dark clock hands, white dial and two stable feet',
'arched_wood_case':'one upright arched mantel clock, broad coloured or wood-finished casing, white dial with dark hands, and flat wide base',
'rectangular_analog':'one upright rectangular tabletop analog clock, broad coloured frame, blank white dial with dark hands, and wide stable flat base',
'carbon_monoxide_alarm':'one compact rectangular carbon monoxide alarm with coloured housing, small inset dark display, front vents and flat wall mounting back',
'security_keypad':'one wall-mounted home security alarm keypad with coloured broad housing, small dark display, compact button pad and flat rear mounting back',
'kitchen_timer':'one small coloured kitchen countdown timer with broad housing, small dark display and flat stable tabletop base',
'portable_rectangular':'one portable rectangular radio, broad coloured cabinet, small dark tuning display, front speaker and flat base',
'rounded_retro':'one rounded-corner retro radio, broad coloured cabinet, small dark tuning display, two knobs and front speaker',
'wood_tabletop':'one compact tabletop radio with broad wood or painted cabinet, front speaker, two small tuning knobs and flat base',
'slim_tablet':'one thin tablet computer in a coloured rear housing, blank dark screen, standing vertically with a small stable integrated kickstand',
'rugged_tablet':'one rugged tablet computer with broad coloured protective housing, blank dark screen and integrated wide stable kickstand',
'convertible_tablet':'one convertible tablet computer with coloured rear housing, blank dark screen and broad folding rear kickstand'}
DESCRIPTIONS.update({
 'co_wide_display':'one wide horizontal carbon monoxide alarm with broad coloured plastic housing, a narrow dark display, small vent slots and flat wall mounting back',
 'co_rounded_vertical':'one upright rounded rectangular carbon monoxide alarm with broad coloured plastic housing, a small dark display, vented front and flat wall mounting back',
 'security_touch_panel':'one wall-mounted security alarm touch panel with broad coloured outer housing, small inset dark touch screen and flat back',
 'security_hinged_cover':'one wall-mounted home security keypad with broad coloured rectangular housing, a partly open integral hinged lid over the small keypad, and a flat rear mounting back',
 'timer_round_digital':'one small round digital kitchen timer with broad coloured casing, narrow dark countdown display, three small buttons and a stable integrated flat foot',
 'timer_clip_on':'one rectangular digital kitchen timer with broad coloured casing, small dark display and a stable integrated rear clip acting as its tabletop foot'
})
COLORS={'black':'matte black','white':'matte white','silver':'brushed neutral silver-grey','gray':'neutral matte grey','red':'deep red','beige':'warm cream beige','wood_brown':'warm walnut wood brown'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[]
    for kind,(forms,colors,dims) in SPECS.items():
        for fi,form in enumerate(forms):
            for color in [colors[fi],colors[(fi+1)%3]]:
                actual=form if kind=='alarm_beep_device' else kind;surface='wall' if actual in ['wall_clock','carbon_monoxide_alarm','security_keypad'] or form in ['over_range','wall_square','compact_rectangular'] else ('ceiling' if form=='ceiling_disc' else 'table')
                if actual=='floorstanding_speaker':surface='floor'
                current=dims.copy()
                if form=='retro_crt':current=[.30,.40,.50]
                if actual=='smoke_detector' and form!='ceiling_disc':current=[.05,.14,.11]
                if actual=='wall_clock':current=[.04,.29,.29]
                prompt='One '+COLORS[color]+' '+DESCRIPTIONS[form]+'. The main structural body and housing are '+COLORS[color]+', visibly covering at least two thirds of the casing. Exactly one complete object, level and correctly oriented, physically realistic construction and proportions, three-quarter product view showing front and right side, slightly elevated camera. Plain seamless light grey background, even soft studio illumination, minimal contact shadow. No text, logos, labels, people, hands, furniture, wires, duplicate objects, cropping or reflection of surroundings.'
                aid='v7_object_'+actual+'_'+form+'_'+color+'_20261012';rows.append({'asset_id':aid,'object_type':actual,'counting_kind':kind,'form_factor':form,'intended_color':color,'prompt':prompt,'seed':2026101200+len(rows),'target_dimensions_m':current,'attachment_surface':surface,'model_route':'flux2_pixal3d_static_v1','profile_origin':'AVEngine candidate_profiles/static_object; form description and crossed colours declared in this plan','max_shape_seeds':3})
    dump(a.out,{'status':'declared_not_generated','requests':rows,'count':len(rows),'crossing':'A: colour1/2; B: colour2/3; C: colour3/1, no texture-only mesh copies counted as independent generations'})
    print('DECLARED',len(rows),'requests in',len(SPECS),'counting groups')
if __name__=='__main__':main()
