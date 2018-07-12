import os
from collections import OrderedDict

from nlp_pneumonia_utils import AnnotatedDocument, Annotation
from intervaltree import IntervalTree


class Evaluator:

    def __init__(self, tp=0, fp=0, tn=None, fn=0):
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn
        #  a dictionary using a doc_name as a key, a list of false negative annotations as a value
        self.fns = OrderedDict()
        #  a dictionary using a doc_name as a key, a list of false positive annotations as a value
        self.fps = OrderedDict()
        pass

    def add_tp(self, n=1):
        self.tp += n

    def add_tn(self, n=1):
        if self.tn is None:
            self.tn = n
        else:
            self.tn += n

    def add_fp(self, n=1):
        self.fp += n

    def add_fn(self, n=1):
        self.fn += n

    def append_fps(self, doc_name, fps):
        if doc_name not in self.fps:
            self.fps[doc_name] = []
        self.fps[doc_name].extend(fps)

    def append_fns(self, doc_name, fns):
        if doc_name not in self.fns:
            self.fns[doc_name] = []
        self.fns[doc_name].extend(fns)

    def get_values(self):
        return self.tp, self.fp, self.tn, self.fn

    def total(self):
        return self.tn + self.tp + self.fp + self.tn

    def get_recall(self):
        return round(100 * self.tp / (self.tp + self.fn)) / 100

    def get_precision(self):
        return round(100 * self.tp / (self.tp + self.fp)) / 100

    def f1(self):
        return round(200 * self.tp / (2 * self.tp + self.fp + self.fn)) / 100

    def get_fns(self) -> dict:
        return self.fns

    def get_fps(self) -> dict:
        return self.fps

    # TODO
    def report(self):
        pass


def group_brat_annotations(lines):
    annotations = {}
    # BRAT FORMAT is:
    # NUMBER[TAB]TYPE[SPACE]START_INDEX[SPACE]END_INDEX[SPACE]SPANNED_TEXT
    for line in lines:
        line = str(line)
        if len(line.strip()) == 0:
            continue
        tab_tokens = line.split('\t')
        space_tokens = tab_tokens[1].split()
        anno = Annotation()
        anno.spanned_text = tab_tokens[-1]
        anno.type = space_tokens[0]
        anno.start_index = int(space_tokens[1])
        anno.end_index = int(space_tokens[-1])
        if (anno.type not in annotations):
            annotations[anno.type] = []
        annotations[anno.type].append(anno)
    return annotations


def docs_reader(dir):
    annotated_doc_map = {}
    for name in sorted(os.listdir(dir)):
        if name.endswith('.txt') or name.endswith('.ann'):
            basename = name.split('.')[0]
            print(name)
            if basename not in annotated_doc_map:
                annotated_doc_map[basename] = AnnotatedDocument()
            anno_doc = annotated_doc_map[basename]
            # handle text and BRAT annotation files (.ann) differently
            if name.endswith('.txt'):
                with open(os.path.join(dir, name)) as f1:
                    anno_doc.text = f1.read()
                f1.close()
            else:
                with open(os.path.join(dir, name)) as f2:
                    anno_doc.grouped_annotations = group_brat_annotations(f2.readlines())
                f2.close()

    return annotated_doc_map


def compare_projects(dir1: str, dir2: str, compare_method: str) -> dict:
    annotated_doc_map1 = docs_reader(dir1)
    annotated_doc_map2 = docs_reader(dir2)
    return compare(annotated_doc_map1, annotated_doc_map2, compare_method)


def compare(annotated_doc_map1: dict, annotated_doc_map2: dict, compare_method='relax') -> dict:
    """
    :param annotated_doc_map1: a dictionary with doc_name as the key, an AnnotatedDocument as the value
    :param annotated_doc_map2:a dictionary with doc_name as the key, an AnnotatedDocument as the value (of reference annotator)
    :param compare_method: "strict" or "relax"
    :return: a dictionary of Evaluators (each Evaluator stores the compared results of one annotation type)
    """
    if len(annotated_doc_map1) != len(annotated_doc_map2):
        raise ValueError("The two input datasets don't have a equal amount of documents.")
        return None
    evaluators = {}
    for doc_name, annotation_doc in annotated_doc_map1.items():
        grouped_annotations2 = annotated_doc_map2[doc_name].grouped_annotations
        if compare_method[0].lower().startswith('s'):
            strict_compare_one_doc(evaluators, doc_name, annotation_doc.grouped_annotations, grouped_annotations2)
        else:
            relax_compare_one_doc(evaluators, doc_name, annotation_doc.grouped_annotations, grouped_annotations2)

    return evaluators


#  consider a match only when the annotations' spans exactly match (both start and end are equal)
#  evaluator, annotations to be compared, reference annotations
def strict_compare_one_doc(evaluators: Evaluator, doc_name: str, grouped_annotations1: [], grouped_annotations2: []):
    for type, annos_list_of_one_type in grouped_annotations1.items():
        if type not in evaluators:
            evaluators[type] = Evaluator()
        evaluator = evaluators[type]

        if type not in grouped_annotations2 or len(grouped_annotations2[type]) == 0:
            evaluator.add_fp(len(annos_list_of_one_type))
            continue

        annos1 = sorted(annos_list_of_one_type, key=lambda x: x.start_index)
        annos2 = sorted(grouped_annotations2[type], key=lambda x: x.start_index)
        
        # Of course, we can try compare each one in a list against all in the other list. 
        # But here is an optimized way        
        # two pointers to track the two lists
        p1 = 0
        p2 = 0
        while p1 < len(annos1) and p2 < len(annos2):
            anno1 = annos1[p1]
            anno2 = annos2[p2]
            if anno1.start_index == anno2.start_index:
                if anno1.end_index == anno2.end_index:
                    evaluator.add_tp()
                else:
                    evaluator.add_fn()
                    evaluator.append_fns(doc_name, [anno2])

                p1 += 1
                p2 += 1
            elif anno1.start_index < anno2.start_index:
                p1 += 1
                evaluator.add_fp()
                evaluator.append_fps(doc_name, [anno1])
            elif anno1.start_index > anno2.start_index:
                p2 += 1
                evaluator.add_fn()
                evaluator.append_fns(doc_name, [anno2])

        if p1 < len(annos1):
            evaluator.add_fp(len(annos1) - p1)
            evaluator.add_fps(annos1[p1:])
        elif p2 < len(annos2):
            evaluator.add_fn(len(annos2) - p2)
            evaluator.add_fps(annos2[p2:])
    pass


def build_interval_tree(annos: []) -> IntervalTree:
    t = IntervalTree()
    for i in range(0, len(annos)):
        begin = annos[i].start_index
        end = annos[i].end_index
        t[begin:end] = i
    return t


def relax_compare_one_doc(evaluators: Evaluator, doc_name: str, grouped_annotations1: [], grouped_annotations2: []):
    for type, annos_list_of_one_type in grouped_annotations1.items():
        if type not in evaluators:
            evaluators[type] = Evaluator()
        evaluator = evaluators[type]

        if type not in grouped_annotations2 or len(grouped_annotations2[type]) == 0:
            evaluator.add_fp(len(annos_list_of_one_type))
            continue
        # we use interval tree to find the overlapped annotations
        # you can try using a similar compare method used in the strict match, but the logic will be way more
        # complicated to implement, when dealing with multiple to one or one to multiple overlaps.
        annos1_tree = build_interval_tree(annos_list_of_one_type)
        overlapped_ids = set()
        for anno2 in grouped_annotations2[type]:
            overlapped = annos1_tree[anno2.start_index:anno2.end_index]
            if len(overlapped) == 0:
                evaluator.add_fn()
                evaluator.append_fns(doc_name, [anno2])
            else:
                evaluator.add_tp()
                overlapped_ids.update([interval.data for interval in overlapped])
        # check what remains in anno1 = false positives
        remain_ids = [i for i in range(0, len(annos_list_of_one_type)) if i not in overlapped_ids]
        if len(remain_ids) > 0:
            evaluator.add_fp(len(remain_ids))
            evaluator.append_fps(doc_name, [annos_list_of_one_type[i] for i in remain_ids])
    pass


