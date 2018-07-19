import pandas as pd
import numpy as np
from xml2mask_2 import camelyon16xml2json
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import os
    xml_dir = 'lesion_annotations'
    df = {'name': [], 'anno_id': [], 'polarity': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': []}
    for root, dirs, files in os.walk(xml_dir):
        for xml_file in sorted(files):
            if not xml_file.endswith('.xml'):
                continue
            name = os.path.splitext(xml_file)[0]

            logging.info('processing {} ...'.format(name))  
            
            json_dict = camelyon16xml2json(os.path.join(root, xml_file))
            for polarity in ['positive', 'negative']:
                for idx, anno_dict in enumerate(json_dict[polarity]):
                    vertices = np.asarray(anno_dict['vertices'], dtype=np.int)
                    xymin = vertices.min(axis=0)
                    xymax = vertices.max(axis=0)
                    df['name'].append(name)
                    df['anno_id'].append(idx)
                    df['polarity'].append(polarity)
                    df['xmin'].append(xymin[0])
                    df['ymin'].append(xymin[1])
                    df['xmax'].append(xymax[0])
                    df['ymax'].append(xymax[1])
    df = pd.DataFrame(df)
    df.to_csv('boundingbox.csv', index=False)
