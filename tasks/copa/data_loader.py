from torchtext.data import Example, Field, Dataset
import torchtext.data as data
import xml.etree.ElementTree as ET


class COPADataset(Dataset):
    """Class for parsing the COPA dataset."""

    def __init__(self, xmlfile, model, encoding="utf-8"):
        text_field = Field(sequential=True, use_vocab=False, include_lengths=True,
                           batch_first=True, pad_token=model.tokenizer.pad_token_id)
        non_seq_field = Field(sequential=False, use_vocab=False, batch_first=True)
        fields = [('ID', non_seq_field), ('event', text_field), ('type_event', non_seq_field),
                  ('hyp_event', text_field), ('hyp_event_ID',  non_seq_field),
                  ('label', non_seq_field)]

        examples = []
        tree = ET.parse(xmlfile)
        # get root element
        root = tree.getroot()

        for item in root.findall('item'):
            id = item.attrib['id']
            event_type_text = item.attrib['asks-for']
            assert(event_type_text in ['cause', 'effect'])
            event_type = (0 if event_type_text == "cause" else 1)

            hyp_label = int(item.attrib['most-plausible-alternative']) - 1
            event, hyp_event_1, hyp_event_2 = None, None, None
            for child in item:
                tokenized_text = model.tokenize(child.text.lower().split())
                if child.tag == "p":
                    event = tokenized_text
                elif child.tag == "a1":
                    hyp_event_1 = tokenized_text
                elif child.tag == "a2":
                    hyp_event_2 = tokenized_text

            for idx, hyp_event in enumerate([hyp_event_1, hyp_event_2]):
                examples.append(
                    Example.fromlist([id, event, event_type, hyp_event, idx,
                                      int(hyp_label == idx)], fields))

        super(COPADataset, self).__init__(examples, fields)

    @staticmethod
    def sort_key(example):
        return len(example.event)

    @classmethod
    def iters(cls, data_path, model, batch_size=32, eval_batch_size=32):
        val, test = COPADataset.splits(
            path=data_path, validation='copa-dev.xml', test='copa-test.xml', model=model)

        (val_iter, ) = data.BucketIterator.splits(
            (val,), batch_size=batch_size,
            sort_within_batch=True, shuffle=False, repeat=False)

        (test_iter, ) = data.BucketIterator.splits(
            (test,), batch_size=eval_batch_size,
            sort_within_batch=True, shuffle=False, repeat=False)

        return (val_iter, test_iter)


if __name__ == '__main__':
    from encoders.pretrained_transformers import Encoder
    encoder = Encoder(cased=False)
    data_path = "/home/shtoshni/Research/causality/tasks/copa/data"
    val_iter, test_iter = COPADataset.iters(data_path, encoder)

    print("Val size:", len(val_iter.data()))
    print("Test size:", len(test_iter.data()))

    for batch_data in val_iter:
        print(batch_data.event[0].shape)
        print(batch_data.hyp_event[0].shape)
        print(batch_data.label.shape)

        text, text_len = batch_data.event
        text_ids = text[0, :text_len[0]].tolist()
        itos = encoder.tokenizer.ids_to_tokens
        sent_tokens = [itos[text_id] for text_id in text_ids]
        sent = ' '.join(sent_tokens)
        # span1 = ' '.join(sent_tokens[batch_data.span1[0, 0]:batch_data.span1[0, 1] + 1])
        print(sent)
        # print(batch_data.span1[0, :])
        # print(span1)
        break
