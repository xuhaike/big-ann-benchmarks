import yaml

def load_runbook(dataset, max_pts, runbook_file):
    with open(runbook_file) as fd:
        runbook = yaml.safe_load(fd)[dataset]
        i=1
        run_list = []
        while i in runbook:
            entry = runbook.get(i)
            if entry['operation'] not in {'insert', 'delete', 'search'}:
                raise Exception('Undefined runbook operation')
            if entry['operation']  in {'insert', 'delete'}:
                if 'start' not in entry:
                    raise Exception('Start not speficied in runbook')
                if 'end' not in entry:
                    raise Exception('End not specified in runbook')
                if entry['start'] < 1 or entry['start'] > max_pts:
                    raise Exception('Start out of range in runbook')
                if entry['end'] < 1 or entry['end'] > max_pts:
                    raise Exception('End out of range in runbook')
            i += 1
            run_list.append(entry)
        return run_list