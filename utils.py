def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def abbreviate(s):
    return ''.join(list(filter(str.isupper, s)))


def to_device(item_list, device):
    for i in range(len(item_list)):
        item_list[i] = item_list[i].to(device)
    return item_list
