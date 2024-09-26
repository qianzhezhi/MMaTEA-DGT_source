from model import SEIR3_faculty

def process(raw):
    try:
        if isinstance(raw, str):
            raw = bytes(raw)
        s = raw.decode('utf-8')
        items = s.strip().split()
        if (len(items) - 1) % 17  != 0:
            print(len(items))
            raise ValueError()

        city = items[0]
        variables = list(map(float, items[1:]))

        return [variables, city]
    except ValueError:
        print(f'Length is wrong')
        return [None, None]
    except :
        print(f'Error processing data: {raw}')
        return [None, None]
    
def evaluate(evaluators, variables, city='default',variant='omicron'):
    # try:
    print(f'process on {city}')
    if city not in evaluators:
        evaluators[city] = SEIR3_faculty(area='default', city=city, variant=variant, max_day=len(variables)//17).make_problem_of_MO()
    return evaluators[city](variables)
    # except:
    #     print(f'Fatal input:  [{variables}, {city}]')
    #     return [float('nan'), float('nan')]