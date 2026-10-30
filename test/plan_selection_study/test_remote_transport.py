"""Transport authentication contract, using mocks only: no network or credentials."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock,patch

import remote_transport as rt


class TTYInput(io.StringIO):
    def isatty(self):
        return True


class TransportTests(unittest.TestCase):
    def argv(self):
        return ['remote_transport.py','--host','example.invalid','--port','24983','--user','root',
                '--known-hosts',str(Path(__file__).parent/'results/hostkeys/cloud_20261028_known_hosts')]

    def test_port_strict_pinned_key_and_tty_password_are_preserved(self):
        client = MagicMock()
        client.get_host_keys.return_value.lookup.return_value = {'ssh-ed25519':object()}
        output = io.StringIO()
        with patch.object(sys,'argv',self.argv()),patch.object(sys,'stdin',TTYInput('{"op":"close"}\n')), \
             patch.object(rt.paramiko,'SSHClient',return_value=client),patch.object(rt.getpass,'getpass',return_value='synthetic-only'), \
             contextlib.redirect_stdout(output):
            rt.main()
        client.get_host_keys.return_value.lookup.assert_called_once_with('[example.invalid]:24983')
        self.assertIsInstance(client.set_missing_host_key_policy.call_args.args[0],rt.paramiko.RejectPolicy)
        self.assertEqual(client.connect.call_args.kwargs['port'],24983)
        self.assertFalse(client.connect.call_args.kwargs['allow_agent'])
        self.assertFalse(client.connect.call_args.kwargs['look_for_keys'])
        self.assertNotIn('synthetic-only',output.getvalue())
        client.close.assert_called_once()

    def test_pipe_input_cannot_read_password_or_connect(self):
        with patch.object(sys,'argv',self.argv()),patch.object(sys,'stdin',io.StringIO()), \
             patch.object(rt.paramiko,'SSHClient') as connect,patch.object(rt.getpass,'getpass') as password, \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                rt.main()
        connect.assert_not_called()
        password.assert_not_called()

    def test_missing_pinned_host_key_cannot_prompt_or_authenticate(self):
        client = MagicMock()
        client.get_host_keys.return_value.lookup.return_value = None
        with patch.object(sys,'argv',self.argv()),patch.object(sys,'stdin',TTYInput()), \
             patch.object(rt.paramiko,'SSHClient',return_value=client),patch.object(rt.getpass,'getpass') as password, \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                rt.main()
        password.assert_not_called()
        client.connect.assert_not_called()


if __name__ == '__main__':
    unittest.main()
