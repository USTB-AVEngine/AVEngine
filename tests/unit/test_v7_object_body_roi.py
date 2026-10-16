"""Functional surfaces may be inset; telephone handsets must remain in housing colour ROIs."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/assets'))
from v7_object_native import functional_exclusion

def test_inset_wall_clock_dial_is_excluded_but_rim_and_back_are_retained():
 q=np.array([[.40,.5,.5],[.90,.95,.5],[.0,.5,.5]])
 assert functional_exclusion({'object_type':'wall_clock','form_factor':'round_analog'},q).tolist()==[True,False,False]

def test_coloured_telephone_handset_is_not_a_keypad():
 q=np.array([[.95,.80,.5],[.4,.3,.5]])
 assert functional_exclusion({'object_type':'desk_telephone'},q).tolist()==[False,True]

def test_inset_speaker_grille_is_excluded_but_back_case_is_retained():
 q=np.array([[.40,.5,.5],[.0,.5,.5]])
 assert functional_exclusion({'object_type':'soundbar'},q).tolist()==[True,False]

def test_tablet_rear_housing_is_retained_even_when_kickstand_extends_behind_it():
 q=np.array([[.8,.5,.5],[.8,.5,.5]])
 assert functional_exclusion({'object_type':'tablet'},q,np.array([False,True])).tolist()==[False,True]

def test_curved_speaker_side_is_retained_when_front_grille_is_inset():
 q=np.array([[.5,.5,.5],[.5,.5,.5]])
 assert functional_exclusion({'object_type':'bookshelf_speaker'},q,np.array([False,True])).tolist()==[False,True]

def test_blender_transparent_jar_is_excluded_from_all_directions():
 q=np.array([[.5,.8,.5],[.5,.8,.5]])
 assert functional_exclusion({'object_type':'blender'},q,np.array([False,True])).tolist()==[True,True]

def test_flat_phone_uses_upward_screen_surface_and_retains_black_edge():
 q=np.array([[.5,.9,.5],[.98,.9,.5]])
 assert functional_exclusion({'object_type':'cellphone','functional_front_axis':'positive-y'},q,np.array([True,True])).tolist()==[True,False]

def test_closed_opaque_keypad_lid_is_body_not_an_exposed_button_region():
 q=np.array([[.9,.4,.5]])
 assert functional_exclusion({'object_type':'security_keypad','form_factor':'keypad_flip_closed'},q,np.array([True])).tolist()==[False]
 assert functional_exclusion({'object_type':'security_keypad','form_factor':'keypad_small_lcd'},q,np.array([True])).tolist()==[True]
