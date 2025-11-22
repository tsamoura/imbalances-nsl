
from pysdd.sdd import SddNode
from typing import List

def createDisjunction(arguments:List[SddNode]):  
    index = 1
    formula = arguments[0]   
    while index < len(arguments):
        formula = formula.disjoin(arguments[index])
        index +=1 
    return formula
    
def createConjunction(arguments:List[SddNode]):  
    index = 1
    formula = arguments[0]
    while index < len(arguments):
        formula = formula.conjoin(arguments[index])
        index +=1 
    return formula
