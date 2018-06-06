#!/usr/bin/python
#
# Cityscapes labels
#

from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    
    # added by Panos
    Label(  'p_limit_20'           ,2000,      7000, 'object'          , 3       , False        , False        , (255,191,  0) ), 
    Label(  'p_limit_30'           ,2001,      7001, 'object'          , 3       , False        , False        , (251,206,177) ), 
    Label(  'p_limit_50'           ,2002,      7002, 'object'          , 3       , False        , False        , (233,214,107) ), 
    Label(  'p_limit_60'           ,2003,      7003, 'object'          , 3       , False        , False        , (253,238,  0) ),
    Label(  'p_limit_70'           ,2004,      7004, 'object'          , 3       , False        , False        , (245,245,220) ),
    Label(  'p_limit_80'           ,2005,      7005, 'object'          , 3       , False        , False        , (240,220,130) ),
    Label(  'o_restriction_ends_80',2006,      7006, 'object'          , 3       , False        , False        , (223,255,  0) ),
    Label(  'p_limit_100'          ,2007,      7007, 'object'          , 3       , False        , False        , (228,208, 10) ),
    Label(  'p_limit_120'          ,2008,      7008, 'object'          , 3       , False        , False        , (255,253,208) ),
    Label(  'p_no_overtaking'      ,2009,      7009, 'object'          , 3       , False        , False        , (194,178,128) ),
    Label(  'p_no_overtaking_truck',2010,      7010, 'object'          , 3       , False        , False        , (238,220,130) ),
    Label(  'd_priority_next_inter',2011,      7011, 'object'          , 3       , False        , False        , (255,215,  0) ),
    Label(  'o_priority_road'      ,2012,      7012, 'object'          , 3       , False        , False        , (212,175, 55) ),
    Label(  'o_give_away'          ,2013,      7013, 'object'          , 3       , False        , False        , (218,165, 32) ),
    Label(  'o_stop'               ,2014,      7014, 'object'          , 3       , False        , False        , (230,168, 23) ),
    Label(  'p_no_traffic_both'    ,2015,      7015, 'object'          , 3       , False        , False        , (248,222,126) ),
    Label(  'p_no_trucks'          ,2016,      7016, 'object'          , 3       , False        , False        , (244,202, 22) ),
    Label(  'o_no_entry'           ,2017,      7017, 'object'          , 3       , False        , False        , (240,230,140) ),
    Label(  'd_danger'             ,2018,      7018, 'object'          , 3       , False        , False        , (255,250,205) ),
    Label(  'd_bend_left'          ,2019,      7019, 'object'          , 3       , False        , False        , (251,236, 93) ),
    Label(  'd_bend_right'         ,2020,      7020, 'object'          , 3       , False        , False        , (255,196, 12) ),
    Label(  'd_bend'               ,2021,      7021, 'object'          , 3       , False        , False        , (255,219, 88) ),
    Label(  'd_uneven_road'        ,2022,      7022, 'object'          , 3       , False        , False        , (250,218, 94) ),
    Label(  'd_slippery_road'      ,2023,      7023, 'object'          , 3       , False        , False        , (255,222,173) ),
    Label(  'd_road_narrows'       ,2024,      7024, 'object'          , 3       , False        , False        , (207,181, 59) ),
    Label(  'd_construction'       ,2025,      7025, 'object'          , 3       , False        , False        , (128,128,  0) ),
    Label(  'd_traffic_signal'     ,2026,      7026, 'object'          , 3       , False        , False        , (255,229,180) ),
    Label(  'd_pedestrian_crossing',2027,      7027, 'object'          , 3       , False        , False        , (244,196, 48) ),
    Label(  'd_school_crossing'    ,2028,      7028, 'object'          , 3       , False        , False        , (255,216,  0) ),
    Label(  'd_cycles_crossing'    ,2029,      7029, 'object'          , 3       , False        , False        , (255,186,  0) ),
    Label(  'd_snow'               ,2030,      7030, 'object'          , 3       , False        , False        , (250,218, 94) ),
    Label(  'd_animals'            ,2031,      7031, 'object'          , 3       , False        , False        , (228,217,111) ),
    Label(  'o_restriction_ends'   ,2032,      7032, 'object'          , 3       , False        , False        , (255,204, 51) ),
    Label(  'm_go_right'           ,2033,      7033, 'object'          , 3       , False        , False        , (243,229,171) ),
    Label(  'm_go_left'            ,2034,      7034, 'object'          , 3       , False        , False        , (255,255,  0) ),
    Label(  'm_go_straight'        ,2035,      7035, 'object'          , 3       , False        , False        , (193,154,107) ),
    Label(  'm_go_right_straight'  ,2036,      7036, 'object'          , 3       , False        , False        , (184,134, 11) ),
    Label(  'm_go_left_straight'   ,2037,      7037, 'object'          , 3       , False        , False        , (255,218,185) ),
    Label(  'm_keep_right'         ,2038,      7038, 'object'          , 3       , False        , False        , (255,203,164) ),
    Label(  'm_keep_left'          ,2039,      7039, 'object'          , 3       , False        , False        , (255,239,  0) ),
    Label(  'm_roundabout'         ,2040,      7040, 'object'          , 3       , False        , False        , (255,211,  0) ),
    Label(  'o_restr_end_ovrtkng'  ,2041,      7041, 'object'          , 3       , False        , False        , (255,255,153) ),
    Label(  'o_restr_end_ovrtkng_t',2042,      7042, 'object'          , 3       , False        , False        , (239,204,  0) ),
    Label(  'o_unknown'            ,2043,       255, 'object'          , 3       , False        , True         , (220,220,  0) ),
    
]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

#--------------------------------------------------------------------------------
# Assure single instance name
#--------------------------------------------------------------------------------

# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None
def assureSingleInstanceName( name ):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of cityscapes labels:")
    print("")
    print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval' ))
    print("    " + ('-' * 98))
    for label in labels:
        print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval ))
    print("")

    print("Example usages:")

    # Map from name to label
    name = 'car'
    id   = name2label[name].id
    print("ID of label '{name}': {id}".format( name=name, id=id ))

    # Map from ID to label
    category = id2label[id].category
    print("Category of label with ID '{id}': {category}".format( id=id, category=category ))

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print("Name of label with trainID '{id}': {name}".format( id=trainId, name=name ))
