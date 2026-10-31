import copy
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from original_compare_guard import parse_nvidia_smi_xml


ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / 'experiments/artifacts/frontier-transfer-20261031/device-query.xml'
UUID = 'GPU-4c8e177a-5a3a-3e0f-61b7-fb666002e0ec'
MINIMAL_XML = '''<?xml version="1.0"?>
<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v13.dtd">
<nvidia_smi_log><attached_gpus>1</attached_gpus>
<gpu id="00000000:00:03.0"><uuid>GPU-4c8e177a-5a3a-3e0f-61b7-fb666002e0ec</uuid>
<pci><pci_bus_id>00000000:00:03.0</pci_bus_id></pci>
<fb_memory_usage><used>1 MiB</used></fb_memory_usage>
<utilization><gpu_util>0 %</gpu_util><memory_util>0 %</memory_util></utilization>
<clocks><sm_clock>210 MHz</sm_clock></clocks>
<temperature><gpu_temp>33 C</gpu_temp></temperature><processes />
</gpu></nvidia_smi_log>'''


class XmlQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = ET.fromstring(MINIMAL_XML)

    def parse(self, root):
        return parse_nvidia_smi_xml(ET.tostring(root, encoding='unicode'), UUID)

    @staticmethod
    def add_process(root, kind, pid='123', name='foreign', memory='32 MiB'):
        process = ET.SubElement(root.find('gpu/processes'), 'process_info')
        for tag, value in [('pid', pid), ('type', kind), ('process_name', name), ('used_memory', memory)]:
            ET.SubElement(process, tag).text = value
        return process

    def test_fixture_and_multi_gpu_host_count(self):
        device, apps = parse_nvidia_smi_xml(MINIMAL_XML, UUID)
        self.assertEqual(device['uuid'], UUID)
        self.assertEqual(device['pci.bus_id'], '00000000:00:03.0')
        self.assertEqual(device['memory.used'], '1')
        self.assertEqual(apps, [])
        root = copy.deepcopy(self.root)
        root.find('attached_gpus').text = '2'
        self.assertEqual(self.parse(root)[1], [])

    @unittest.skipUnless(FIXTURE.is_file(), 'locally captured XML is unavailable')
    def test_captured_real_fixture(self):
        device, apps = parse_nvidia_smi_xml(FIXTURE.read_text(encoding='utf-8'), UUID)
        self.assertEqual(device['uuid'], UUID)
        self.assertEqual(apps, [])

    def test_compute_and_combined_include_foreign_graphics_excluded(self):
        root = copy.deepcopy(self.root)
        self.add_process(root, 'C', '321', 'foreign-cuda', '12 MiB')
        self.add_process(root, 'G', '322', 'desktop', 'N/A')
        self.add_process(root, 'C+G', '323', 'foreign-combined', '34 MiB')
        _, apps = self.parse(root)
        self.assertEqual([app['pid'] for app in apps], ['321', '323'])
        self.assertEqual(apps[0], dict(gpu_uuid=UUID, pid='321', process_name='foreign-cuda', used_gpu_memory='12'))

    def test_only_graphics_is_not_compute(self):
        root = copy.deepcopy(self.root)
        self.add_process(root, 'G')
        self.assertEqual(self.parse(root)[1], [])

    def test_missing_and_ambiguous_process_info_fails_closed(self):
        for bad_type in ('N/A', 'C/G', '', 'unknown'):
            with self.subTest(bad_type=bad_type):
                root = copy.deepcopy(self.root)
                self.add_process(root, bad_type)
                with self.assertRaises(ValueError):
                    self.parse(root)
        for field, value in [('pid', '0'), ('pid', 'no'), ('used_memory', 'N/A'), ('used_memory', '-1 MiB')]:
            with self.subTest(field=field, value=value):
                root = copy.deepcopy(self.root)
                process = self.add_process(root, 'C')
                process.find(field).text = value
                with self.assertRaises(ValueError):
                    self.parse(root)
        root = copy.deepcopy(self.root)
        root.find('gpu').remove(root.find('gpu/processes'))
        with self.assertRaises(ValueError):
            self.parse(root)
        root = copy.deepcopy(self.root)
        root.find('gpu/processes').text = 'N/A'
        with self.assertRaises(ValueError):
            self.parse(root)

    def test_unavailable_process_label_does_not_hide_compute_pid(self):
        for name in ('', 'N/A'):
            root=copy.deepcopy(self.root)
            process=self.add_process(root,'C')
            process.find('process_name').text=name
            _,apps=self.parse(root)
            self.assertEqual(len(apps),1)
            self.assertEqual(apps[0]['pid'],process.find('pid').text)
            self.assertEqual(apps[0]['process_name'],'N/A')

    def test_gpu_identity_fields_and_counts_fail_closed(self):
        for path, value in [('attached_gpus', 'N/A'), ('attached_gpus', '0'),
                            ('gpu/uuid', 'GPU-other'), ('gpu/pci/pci_bus_id', 'N/A'),
                            ('gpu/fb_memory_usage/used', 'NaN MiB'),
                            ('gpu/utilization/gpu_util', '-1 %'),
                            ('gpu/utilization/memory_util', '101 %'),
                            ('gpu/clocks/sm_clock', 'N/A'),
                            ('gpu/temperature/gpu_temp', 'inf C')]:
            with self.subTest(path=path):
                root = copy.deepcopy(self.root)
                root.find(path).text = value
                with self.assertRaises(ValueError):
                    self.parse(root)
        root = copy.deepcopy(self.root)
        root.append(copy.deepcopy(root.find('gpu')))
        with self.assertRaises(ValueError):
            self.parse(root)
        root = copy.deepcopy(self.root)
        root.find('gpu').set('id', '00000000:00:04.0')
        with self.assertRaises(ValueError):
            self.parse(root)
        root = copy.deepcopy(self.root)
        root.find('gpu/utilization').append(copy.deepcopy(root.find('gpu/utilization/gpu_util')))
        with self.assertRaises(ValueError):
            self.parse(root)

    def test_malformed_and_entity_xml_rejected(self):
        for text in ('<broken', '<!DOCTYPE nvidia_smi_log [<!ENTITY e "bad">]><nvidia_smi_log/>'):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_nvidia_smi_xml(text, UUID)


if __name__ == '__main__':
    unittest.main()
