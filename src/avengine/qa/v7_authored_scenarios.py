"""Author-written daily semantic catalogue; no inferred or asset-derived labels.

The 180 scenarios use 15 conversational functions in 12 concrete settings.
Variation changes the requested action, object, route, time or message, rather
than merely renaming a person. English speech follows the VCTK convention.
"""
from __future__ import annotations
import hashlib
import json

SETTINGS = (
    ('breakfast_kitchen', '早餐厨房', 'while we prepare breakfast'),
    ('apartment_hall', '公寓门厅', 'before we leave the apartment'),
    ('office_desk', '办公室桌旁', 'before the morning meeting'),
    ('station_entrance', '车站入口', 'before the train arrives'),
    ('school_lobby', '学校大厅', 'before the afternoon class'),
    ('community_room', '社区活动室', 'before the community event'),
    ('corner_shop', '街角商店', 'before the shop closes'),
    ('garden_gate', '花园门口', 'before we go into the garden'),
    ('clinic_waiting', '诊所等候区', 'before the appointment'),
    ('hotel_reception', '旅馆前台', 'before the guests check in'),
    ('rehearsal_room', '排练室', 'before the rehearsal starts'),
    ('home_living_room', '家中客厅', 'before our visitors arrive'),
)
SPOKEN_CONTEXT = (
    'We are getting breakfast ready.', 'We are about to leave the apartment.',
    'The morning meeting is coming up.', 'Our train will arrive soon.',
    'The afternoon class will start soon.', 'The community event is starting soon.',
    'The shop is about to close.', 'We are about to go into the garden.',
    'Our appointment will begin soon.', 'The guests will check in soon.',
    'The rehearsal is about to start.', 'Our visitors will arrive soon.',
)
# Every row is an authored (English meaning, Chinese meaning) alternative.
OBJECTS = (
    (('a clean mug','干净杯子'),('a tea spoon','茶匙'),('a breakfast plate','早餐盘'),('a paper napkin','餐巾纸')),
    (('the front door key','前门钥匙'),('my blue umbrella','蓝色雨伞'),('the grocery bag','购物袋'),('my bike helmet','自行车头盔')),
    (('a black pen','黑色笔'),('the meeting folder','会议文件夹'),('a phone charger','手机充电器'),('a blank notebook','空白笔记本')),
    (('my train ticket','火车票'),('a station map','车站地图'),('a bottle of water','瓶装水'),('the luggage tag','行李牌')),
    (('a library card','借书卡'),('a math workbook','数学练习册'),('a pencil case','笔袋'),('a permission form','许可表')),
    (('a folding chair','折叠椅'),('the guest list','来宾名单'),('a name badge','姓名牌'),('a roll of tape','胶带')),
    (('a shopping basket','购物篮'),('a receipt envelope','收据袋'),('a carton of milk','牛奶盒'),('a reusable bag','环保袋')),
    (('a watering can','浇水壶'),('a pair of gloves','手套'),('a garden trowel','园艺铲'),('a seed packet','种子包')),
    (('my appointment card','预约卡'),('a tissue box','纸巾盒'),('my insurance card','保险卡'),('the intake form','登记表')),
    (('a room key card','房卡'),('a spare towel','备用毛巾'),('a city guide','城市指南'),('a luggage trolley','行李推车')),
    (('the sheet music','乐谱'),('a guitar pick','吉他拨片'),('a music stand','谱架'),('a spare microphone cable','备用麦克风线')),
    (('the television remote','电视遥控器'),('a sofa cushion','沙发靠垫'),('a serving tray','托盘'),('a coaster','杯垫')),
)
ROUTES = (
    (('the cafe past the bakery','经过面包店去咖啡馆'),('the bus stop across the bridge','过桥去公交站'),('the pharmacy beside the bank','去银行旁药房'),('the park through the side gate','穿侧门去公园')),
    (('the mailbox downstairs','下楼去信箱'),('the laundry room along the corridor','沿走廊去洗衣房'),('the recycling bins behind the building','去楼后回收桶'),('the garage down the ramp','下坡道去车库')),
    (('the printer room on the right','去右侧打印室'),('the meeting room on the second floor','去二楼会议室'),('the cafeteria beyond the lift','去电梯后食堂'),('the exit at the end of the hall','去走廊尽头出口')),
    (('platform two through the tunnel','穿隧道去二号站台'),('the ticket desk to the left','去左侧售票处'),('the taxi rank outside the east exit','去东出口外出租车站'),('the lockers beside the stairs','去楼梯旁寄存柜')),
    (('the science lab upstairs','上楼去科学实验室'),('the library past the courtyard','穿庭院去图书馆'),('the gym behind the main hall','去大厅后体育馆'),('the music room along the west corridor','沿西走廊去音乐室')),
    (('the registration table near the door','去门旁登记桌'),('the kitchen behind the stage','去舞台后厨房'),('the toilets beside the noticeboard','去公告板旁厕所'),('the coat rack by the window','去窗边衣帽架')),
    (('the fruit aisle near the entrance','去入口附近水果区'),('the bread shelf beside the freezer','去冰柜旁面包架'),('the checkout at the back','去后方收银处'),('the parcel desk next door','去隔壁包裹柜台')),
    (('the greenhouse along the gravel path','沿碎石路去温室'),('the pond beyond the wooden bench','去木凳后池塘'),('the tool shed beside the hedge','去树篱旁工具棚'),('the compost bin behind the apple tree','去苹果树后堆肥箱')),
    (('the reception desk through the glass door','穿玻璃门去接待台'),('the blood test room on the first floor','去一楼验血室'),('the waiting room along the blue line','沿蓝线去等候室'),('the pharmacy outside the north exit','去北出口外药房')),
    (('the breakfast room beyond the lounge','去休息区后早餐厅'),('the lift behind reception','去前台后电梯'),('the car park through the rear door','穿后门去停车场'),('the bus stop opposite the hotel','去旅馆对面公交站')),
    (('the recording booth at the back','去后方录音间'),('the instrument store through the small door','穿小门去乐器库'),('the green room upstairs','上楼去休息室'),('the stage through the red curtain','穿红幕去舞台')),
    (('the guest room upstairs','上楼去客房'),('the bathroom beside the kitchen','去厨房旁浴室'),('the balcony through the living room','穿客厅去阳台'),('the spare blankets in the hall cupboard','去门厅柜取备用毯')),
)
TASKS = (
    (('toast the bread','烤面包'),('slice the tomatoes','切番茄'),('boil the eggs','煮鸡蛋'),('set the breakfast table','摆早餐餐具')),
    (('hang up the coats','挂好外套'),('wipe the muddy shoes','擦泥鞋'),('sweep the doorway','扫门口'),('sort the letters','整理信件')),
    (('print the agenda','打印议程'),('charge the conference phone','给会议电话充电'),('refill the paper tray','补打印纸'),('connect the projector','连接投影仪')),
    (('check the departure board','查发车屏'),('label the suitcase','给行李贴牌'),('refill the travel bottle','装满旅行水瓶'),('fold the station map','折车站地图')),
    (('collect the homework','收作业'),('erase the whiteboard','擦白板'),('stack the library books','叠图书'),('arrange the art supplies','摆美术用品')),
    (('unfold the chairs','展开椅子'),('count the name badges','点姓名牌'),('fill the water jug','装水壶'),('clear the stage entrance','清舞台入口')),
    (('check the milk dates','查牛奶日期'),('count the bread rolls','数小面包'),('wipe the checkout counter','擦收银台'),('fold the reusable bags','叠环保袋')),
    (('water the seedlings','浇幼苗'),('collect the fallen leaves','收落叶'),('clean the garden tools','洗园艺工具'),('fill the bird feeder','装鸟食')),
    (('complete the intake form','填登记表'),('put the cards in an envelope','把卡放信封'),('refill the tissue holder','补纸巾'),('clean the waiting chairs','清等候椅')),
    (('prepare the welcome cards','准备欢迎卡'),('fold the spare towels','叠备用毛巾'),('charge the desk tablet','给前台平板充电'),('sort the room keys','整理房卡')),
    (('tune the acoustic guitar','给木吉他调音'),('arrange the music stands','摆谱架'),('replace the microphone battery','换麦克风电池'),('coil the spare cables','卷备用线')),
    (('put away the toys','收玩具'),('wipe the coffee table','擦茶几'),('fold the sofa blanket','叠沙发毯'),('open the curtains','拉开窗帘')),
)
TIMES = (
    ('seven fifteen','七点十五'),('eight thirty','八点三十'),('nine forty-five','九点四十五'),
    ('ten twenty','十点二十'),('eleven fifty','十一点五十'),('twelve ten','十二点十分'),
    ('one thirty-five','一点三十五'),('two forty','两点四十'),('three twenty-five','三点二十五'),
    ('four fifty-five','四点五十五'),('five ten','五点十分'),('six forty','六点四十'),
)
PROBLEMS = (
    (('the toaster will not heat','烤面包机不加热'),('the kettle leaks','水壶漏水'),('the fridge door will not shut','冰箱门关不上'),('the tap keeps dripping','水龙头滴水')),
    (('the hallway light is broken','门厅灯坏了'),('the front lock is stiff','前门锁发涩'),('the lift has stopped','电梯停了'),('the intercom is too quiet','对讲机太轻')),
    (('the printer is jammed','打印机卡纸'),('the projector image is blurry','投影模糊'),('the desk chair is loose','办公椅松动'),('the network keeps dropping','网络断连')),
    (('the ticket machine has frozen','售票机死机'),('the platform clock is wrong','站台钟不准'),('the luggage wheel is stuck','行李轮卡住'),('the departure screen is blank','发车屏空白')),
    (('the classroom is too cold','教室太冷'),('the workbook has missing pages','练习册缺页'),('the locker will not open','柜门打不开'),('the marker has dried out','白板笔干了')),
    (('the speaker cable is loose','音箱线松了'),('the stage step is slippery','舞台台阶滑'),('the folding chair is unstable','折叠椅不稳'),('the water jug is cracked','水壶裂了')),
    (('the receipt is incorrect','收据有误'),('the milk carton is leaking','牛奶盒漏了'),('the checkout queue is blocked','收银队伍堵了'),('the freezer is too warm','冰柜太热')),
    (('the hose has a hole','水管有洞'),('the gate hinge is rusty','门铰生锈'),('the bird feeder is empty','鸟食盒空了'),('the wheelbarrow tyre is flat','独轮车胎瘪了')),
    (('the waiting chair is broken','等候椅坏了'),('the appointment date is wrong','预约日期有误'),('the card reader will not work','读卡器失灵'),('the room is too noisy','房间太吵')),
    (('the room key will not work','房卡失灵'),('the towel is stained','毛巾有污点'),('the room heater is noisy','暖气太吵'),('the suitcase handle is stuck','行李把手卡住')),
    (('the guitar string has snapped','吉他弦断了'),('the music stand keeps falling','谱架总倒'),('the microphone has no power','麦克风没电'),('the piano bench is uneven','琴凳不平')),
    (('the remote has no batteries','遥控器没电池'),('the curtain rail is stuck','窗帘轨卡了'),('the sofa cushion is torn','靠垫破了'),('the doorbell is too loud','门铃太响')),
)
FOODS = (
    (('porridge with honey','蜂蜜粥'),('toast with jam','果酱吐司'),('plain yoghurt','原味酸奶'),('scrambled eggs','炒蛋')),
    (('vegetable soup','蔬菜汤'),('a cheese sandwich','奶酪三明治'),('a bean salad','豆沙拉'),('a baked potato','烤土豆')),
    (('coffee without sugar','无糖咖啡'),('tea with milk','加奶茶'),('still water','白水'),('orange juice','橙汁')),
    (('a banana','香蕉'),('an apple','苹果'),('a rice cake','米饼'),('a cereal bar','谷物棒')),
    (('a vegetable wrap','蔬菜卷'),('a chicken sandwich','鸡肉三明治'),('tomato pasta','番茄意面'),('lentil soup','扁豆汤')),
    (('mint tea','薄荷茶'),('hot chocolate','热巧克力'),('lemon water','柠檬水'),('black coffee','黑咖啡')),
    (('wholemeal bread','全麦面包'),('brown rice','糙米'),('plain crackers','原味薄脆饼'),('rolled oats','燕麦片')),
    (('a pear','梨'),('a peach','桃'),('strawberries','草莓'),('grapes','葡萄')),
    (('decaffeinated coffee','无咖啡因咖啡'),('warm water','温水'),('unsweetened tea','无糖茶'),('apple juice','苹果汁')),
    (('a mushroom omelette','蘑菇煎蛋'),('pancakes','煎饼'),('a fruit bowl','水果碗'),('a bagel','贝果')),
    (('herbal tea','花草茶'),('sparkling water','气泡水'),('a mango smoothie','芒果冰沙'),('cold milk','冷牛奶')),
    (('a vegetable pizza','蔬菜披萨'),('noodle soup','面汤'),('roast chicken','烤鸡'),('steamed rice','蒸米饭')),
)

FUNCTIONS = (
    ('request_item', '想要的东西', 'Which item does this person ask you to bring?', '这个人请你拿什么？'),
    ('ask_directions', '询问的路线', 'Which route does this person ask about?', '这个人在问去哪里的路线？'),
    ('arrange_time', '约定时间', 'When does this person suggest meeting?', '这个人提议何时见面？'),
    ('kitchen_instruction', '交代的事情', 'What task does this person ask you to do?', '这个人请你做什么事？'),
    ('phone_message', '电话里传达的消息', 'What does the caller say they will do?', '打电话的人说自己将做什么？'),
    ('megaphone_notice', '喇叭通知的事项', 'What does the announcer ask people to do?', '拿喇叭的人通知大家做什么？'),
    ('complaint', '抱怨的问题', 'What problem does this person report?', '这个人抱怨什么问题？'),
    ('greeting', '打招呼的意思', 'What is the purpose of this greeting?', '这句招呼表达什么意思？'),
    ('shopping', '购买的东西', 'Which item does this person want to buy?', '这个人想买什么？'),
    ('reminder', '提醒的行动', 'What action does this person remind you to take?', '这个人提醒你做什么？'),
    ('lost_item', '寻找的东西', 'Which missing item is this person looking for?', '这个人在找丢失的什么东西？'),
    ('chore', '承担的任务', 'What task does this person offer to take?', '这个人愿意承担什么任务？'),
    ('meeting_route', '会合地点', 'Where does this person suggest meeting?', '这个人提议在哪里会合？'),
    ('help_offer', '提供的帮助', 'How does this person offer to help?', '这个人愿意怎样帮忙？'),
    ('food_preference', '饮食选择', 'Which food or drink does this person choose?', '这个人选什么食物或饮料？'),
)

def _alternatives(function, context_index):
    i = context_index
    if function in {'request_item','shopping','lost_item'}:
        values = OBJECTS[i]
        patterns = {'request_item':"Please bring me {value}.",
                    'shopping':"I would like to buy {value}.",
                    'lost_item':"I cannot find {value}. Please help me look for it."}
        return [(en, zh, patterns[function].format(value=en)) for en, zh in values]
    if function in {'ask_directions','meeting_route'}:
        pattern = "How do I get to {value}?" if function == 'ask_directions' else "Let us meet at {value}."
        return [(en, zh, pattern.format(value=en)) for en, zh in ROUTES[i]]
    if function == 'arrange_time':
        purposes = ('breakfast','the move','the meeting','the train','class','the event',
                    'shopping','gardening','the appointment','check-in','rehearsal','the visit')
        return [(en, zh, "Let us meet for "+purposes[i]+" at "+en+".")
                for en, zh in (TIMES[(i+j*3)%12] for j in range(4))]
    if function in {'kitchen_instruction','phone_message','megaphone_notice','reminder','chore','help_offer'}:
        patterns = {'kitchen_instruction':"Please {value}.",
                    'phone_message':"I will {value} before I arrive.",
                    'megaphone_notice':"Attention! Please {value}.",
                    'reminder':"Remember to {value}.",
                    'chore':"I will {value}.",
                    'help_offer':"I can help you {value}."}
        return [(en, zh, patterns[function].format(value=en)) for en, zh in TASKS[i]]
    if function == 'complaint':
        return [(en, zh, en.capitalize()+".") for en, zh in PROBLEMS[i]]
    if function == 'greeting':
        # Four intents differ, not merely four names or vocal presets.
        spots = ('the kitchen','the apartment','the office','the station','the school','the community room',
                 'the shop','the garden','the clinic','the hotel','the rehearsal room','our home')
        place = spots[i]
        values = (('welcome a newcomer','欢迎新来的人',f"Welcome to {place}. It is your first visit."),
                  ('greet a returning friend','问候再次来访的朋友',f"Hello again. Good to see you back in {place}."),
                  ('thank someone for coming','感谢对方到来',f"Thank you for coming to {place}."),
                  ('say goodbye for the day','当天告别',f"I am leaving {place}. Goodbye for today."))
        return list(values)
    if function == 'food_preference':
        return [(en, zh, "For my meal, I would choose "+en+".") for en, zh in FOODS[i]]
    raise ValueError(function)

def build_catalog(reference_plans=()):
    references = list(reference_plans)
    by_gender = {gender:[r for r in references if r['gender']==gender and r.get('split')=='train']
                 for gender in ('female','male')}
    scenarios = []
    for f, slot, q_en, q_zh in FUNCTIONS:
        for i, (context, zh_context, context_text) in enumerate(SETTINGS):
            number = len(scenarios)
            role_count = 2+number%3
            sid = 'daily_'+f+'_'+context
            choices = _alternatives(f, i)
            # Content alternatives are authored; short complete turns leave room for native role schedules.
            utterances = [{'answer':'meaning_'+str(j+1), 'label_en':en, 'label_zh':zh,
                           'text':text,
                           'aliases':[en,zh], 'critical_semantic_slots':([('first visit', 'see you back', 'Thank you for coming', 'leaving')[j]] if f=='greeting' else [en])}
                          for j,(en,zh,text) in enumerate(choices)]
            # This permutation is independent of appearance and voice.
            offset = int(hashlib.sha256(sid.encode()).hexdigest()[:8],16)%4
            assigned = [(offset+j)%4 for j in range(role_count)]
            roles = []
            for j, option in enumerate(assigned):
                gender = ('female','male')[(number+j)%2]
                refs = by_gender[gender]
                ref = refs[(number+j)%len(refs)] if refs else None
                kind = 'phone_call' if f=='phone_message' else 'megaphone_speaker' if f=='megaphone_notice' else 'standing_human'
                roles.append({'role_id':'role_'+str(j+1), 'role_description_zh':'本场景第'+str(j+1)+'个参与者',
                  'meaning_answer':utterances[option]['answer'], 'transcript':utterances[option]['text'],
                  'required_visual_gender':gender,'voice_gender_plan':gender, 'required_asset_capability':kind,
                  'voice_reference_plan':ref,'voice_reference_actual_used':None,
                  'synthesis_config':{'engine':'CosyVoice2','model_id':'FunAudioLLM/CosyVoice2-0.5B',
                    'mode':'zero_shot','language':'en','seed':10000+number*4+j,
                    'complete_utterance_required':True,'native_window_fit_required':True,'speed':1.0,'stream':False,'device':'cpu',
                    'trim_or_repeat_permitted':False,'automatic_ASR_required':True,
                    'model_license':'pending_local_weight_license_receipt','actual_waveform':None},
                  'role_evidence_required':['physical_actor_id_private','instance_id_private','appearance_descriptor',
                     'measured_complete_utterance_timeline','native_FOA_emitter_attachment','voice_visual_gender'],
                  'text_license':'project_authored_text','reference_license':'CC-BY-4.0' if ref else 'pending_reference',
                  'status':'authored_not_synthesized'})
            scenarios.append({'id':sid,'slot':slot,'context_zh':zh_context,'conversational_function':f,
                 'question_en':q_en,'question_zh':q_zh,
                 'question_zh_alternate':'请根据这位人物所说的内容，回答：'+q_zh,
                 'utterances':utterances,'roles':roles,'role_count':role_count,
                 'wording_assignment_depends_on_answer':False,'language':'en','status':'authored_not_synthesized',
                 'bindings_required':['QA-26_appearance_to_meaning','QA-27_meaning_to_appearance','QA-28_meaning_to_sector']})
    return {'schema':'v7_daily_author_dialogues_v2','author':'project_author','scenarios':scenarios,
       'language_policy':{'spoken_language':'en','english_fraction':1.0,'chinese_fraction':0.0,
                          'reason':'follows existing English VCTK speech and semantic scripts; Chinese question text stays supported'},
       'authoring_structure':{'conversational_functions':15,'settings_per_function':12,
                             'scenarios':180,'not_180_independent_dialogue_templates':True},
       'actual_synthesized_utterances':0,'actual_TTS_reference_speaker_ids':[],
       'planned_reference_speaker_ids':sorted({r['speaker_id'] for r in references}),
       'formal_native_semantic_questions':0}

def validate_catalog(catalog):
    ids=set();utterance_texts=set();role_totals={2:0,3:0,4:0};errors=[];used_meanings=0
    for s in catalog['scenarios']:
        if s['id'] in ids:errors.append(s['id']+':duplicate_scenario')
        ids.add(s['id'])
        u=s['utterances'];answers=[x['answer'] for x in u]
        if len(answers)!=len(set(answers)) or len({x['label_en'] for x in u})!=len(u):
            errors.append(s['id']+':ambiguous_answer_domain')
        for x in u:
            if x['text'] in utterance_texts:errors.append(s['id']+':duplicate_full_spoken_text')
            utterance_texts.add(x['text'])
            if not x['critical_semantic_slots'] or not all(k.lower() in x['text'].lower() for k in x['critical_semantic_slots']):
                errors.append(s['id']+':critical_meaning_missing_from_transcript')
        roles=s['roles'];assigned=[r['meaning_answer'] for r in roles]
        if len(roles) not in role_totals or len(roles)!=len(set(assigned)):
            errors.append(s['id']+':roles_not_distinct_2_to_4')
        else:role_totals[len(roles)]+=1
        for r in roles:
            exact=[x for x in u if x['answer']==r['meaning_answer']]
            if len(exact)!=1 or exact[0]['text']!=r['transcript']:errors.append(s['id']+':role_script_mismatch')
            if r['voice_gender_plan']!=r['required_visual_gender']:errors.append(s['id']+':voice_visual_gender_plan_mismatch')
            ref=r.get('voice_reference_plan')
            if ref and (ref['gender']!=r['voice_gender_plan'] or ref.get('split')!='train'
                        or ref.get('license_chain_status')!='pass'):
                errors.append(s['id']+':reference_plan_not_licensed_training_gender')
        used_meanings+=len(assigned)
        if s.get('wording_assignment_depends_on_answer') is not False:
            errors.append(s['id']+':wording_answer_dependence')
        for q in (s['question_zh'],s['question_zh_alternate'],s['question_en']):
            if any(x['label_en'].lower() in q.lower() or x['label_zh'] in q for x in u):
                errors.append(s['id']+':answer_label_in_generic_question')
    return {'status':'pass' if not errors else 'blocked','scenario_count':len(ids),
            'author_options':len(utterance_texts),'assigned_role_utterances':used_meanings,
            'scenarios_by_role_count':role_totals,'errors':errors,
            'actual_synthesized_utterances':catalog['actual_synthesized_utterances'],
            'actual_TTS_reference_speaker_ids':catalog['actual_TTS_reference_speaker_ids'],
            'checks':'unambiguous author meanings, role scripts, distinct actors, licence/gender plan, wording interchange',
            'scope_limit':'author design audit, no claim that ASR or listeners verified unsynthesized audio'}
