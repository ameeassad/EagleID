

class AKSkeletonCategory:
    """
    Handles the extraction and organization of skeleton keypoints and connections from the COCO annotations.

    Args:
        coco_data (dict): COCO-style annotations.

    Attributes:
        connections (list): List of connections (limbs) between keypoints.
        joint_names (list): Names of the keypoints (joints).
    """
    def __init__(self, coco_data=None):
        if coco_data is None:
            coco_data = { 
                "categories" : [{
                    "supercategory" : "bird",
                    "id" : 1,
                    "name" : "eagle",
                }]
            }
        for cat in coco_data['categories']:
            if not cat.get('keypoints'):
                cat['keypoints'] = [
                    "Head_Mid_Top",
                    "Eye_Left",
                    "Eye_Right",
                    "Mouth_Front_Top",
                    "Mouth_Back_Left",
                    "Mouth_Back_Right",
                    "Mouth_Front_Bottom",
                    "Shoulder_Left",
                    "Shoulder_Right",
                    "Elbow_Left",
                    "Elbow_Right",
                    "Wrist_Left",
                    "Wrist_Right",
                    "Torso_Mid_Back",
                    "Hip_Left",
                    "Hip_Right",
                    "Knee_Left",
                    "Knee_Right",
                    "Ankle_Left",
                    "Ankle_Right",
                    "Tail_Top_Back",
                    "Tail_Mid_Back",
                    "Tail_End_Back"
                ]
                cat['skeleton'] = [
                    [2,1],
                    [3,1],
                    [4,5],
                    [4,6],
                    [7,5],
                    [7,6],
                    [1,14],
                    [14,21],
                    [21,22],
                    [22,23],
                    [1,8],
                    [1,9],
                    [8,10],
                    [9,11],
                    [10,12],
                    [11,13],
                    [21,15],
                    [21,16],
                    [15,17],
                    [16,18],
                    [17,19],
                    [18,20]
                ]
                self.connections = cat['skeleton']
                self.joint_names = cat['keypoints']
        self.coco_data = coco_data

    def __call__(self):
        return self.coco_data

    def get_updated_categories(self):
        return self.coco_data['categories']
    
    def get_connections(self):
        return self.connections
    
    def get_joint_names(self):  
        return self.joint_names