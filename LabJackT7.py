# -*- coding: utf-8 -*-
#
# This file is part of the LabJackT7 project
#
#
#
# Distributed under the terms of the MIT license.
# See LICENSE.txt for more info.

""" LabJackT7 analog input

Configure and read analog inputs on LabJackT7.
Supports single-shot and stream measurements.
"""

# PyTango imports
import tango
from tango import DebugIt
from tango.server import run
from tango.server import Device
from tango.server import attribute, command
from tango.server import device_property
from tango import AttrQuality, DispLevel, DevState
from tango import AttrWriteType, PipeWriteType

# Additional import
# PROTECTED REGION ID(LabJackT7.additionnal_import) ENABLED START #
import asyncio
from labjack import ljm
import numpy as np

# PROTECTED REGION END #    //  LabJackT7.additionnal_import

__all__ = ["LabJackT7", "main"]


class LabJackT7(Device):
    """
    Configure and read analog inputs on LabJackT7.
    Supports single-shot and stream measurements.

    **Properties:**

    - Device Property
        connection_type
            - ``USB``, ``ETHERNET`` or ``ANY``
            - Type:'DevString'
        IPaddress
            - IP address of LabJackT7 in case ethernet is being used
            - Type:'DevString'
    """

    # PROTECTED REGION ID(LabJackT7.class_variable) ENABLED START #
    green_mode = tango.GreenMode.Asyncio
    # PROTECTED REGION END #    //  LabJackT7.class_variable

    # -----------------
    # Device Properties
    # -----------------

    connection_type = device_property(dtype="DevString", default_value="ETHERNET")

    IPaddress = device_property(dtype="DevString", default_value="ANY")

    # ----------
    # Attributes
    # ----------

    AIN0 = attribute(
        dtype="DevDouble",
        unit="V",
    )

    AIN1 = attribute(
        dtype="DevDouble",
        unit="V",
    )

    AIN2 = attribute(
        dtype="DevDouble",
        unit="V",
    )

    AIN3 = attribute(
        dtype="DevDouble",
        unit="V",
    )

    buffer_scanrate = attribute(
        dtype="DevLong",
        access=AttrWriteType.READ_WRITE,
        unit="Hz",
        doc="acquisition rate for buffered measurements",
    )

    buffer_size = attribute(
        dtype="DevLong",
        access=AttrWriteType.READ_WRITE,
        doc="number of samples to read in buffered measurement",
    )

    arrayAIN0 = attribute(
        dtype=("DevDouble",),
        max_dim_x=100000,
        unit="V",
    )

    arrayAIN1 = attribute(
        dtype=("DevDouble",),
        max_dim_x=100000,
        unit="V",
    )

    arrayAIN2 = attribute(
        dtype=("DevDouble",),
        max_dim_x=100000,
        unit="V",
    )

    arrayAIN3 = attribute(
        dtype=("DevDouble",),
        max_dim_x=100000,
        unit="V",
    )

    # ---------------
    # General methods
    # ---------------

    async def init_device(self):
        """Initialises the attributes and properties of the LabJackT7."""
        await Device.init_device(self)
        # PROTECTED REGION ID(LabJackT7.init_device) ENABLED START #
        self._buffer_scanrate = 1000
        self._buffer_size = 10
        self._databuffer = np.zeros((4, self._buffer_size))
        self._stop_buffer = False
        await self.connect()

    @command(dtype_in=int, doc_in="channel number (0-3)")
    def read_AIN(self, channel):
        if channel > 3:
            raise ValueError("Allowed channels: 0-3")
        value = ljm.eReadName(self.handle, f"AIN{channel}")
        return float(value)

    def is_read_AIN_allowed(self):
        return self.get_state() not in [DevState.RUNNING]

    def set_params_from_dict(self, params: dict):
        """Set labjack configuration from dictionary."""
        ljm.eWriteNames(self.handle, len(params), list(params), list(params.values()))

    async def stream_worker(self):
        """Read stream data asynchronously."""
        channelnames = [f"AIN{ch}" for ch in range(4)]
        numAddresses = len(channelnames)
        channel_ids = ljm.namesToAddresses(numAddresses, channelnames)[0]
        # aim for 5 updates/second, but no less than 10
        scansPerRead = max(10, self._buffer_scanrate // 5)
        n_reads = int(np.ceil(self._buffer_size / scansPerRead))
        self._databuffer = np.nan * np.zeros((numAddresses, scansPerRead * n_reads))

        self.set_state(DevState.RUNNING)
        scanrate = ljm.eStreamStart(
            self.handle, scansPerRead, numAddresses, channel_ids, self._buffer_scanrate
        )
        print(f"stream_worker {scanrate=}", file=self.log_debug)

        for i in range(n_reads):
            chunk, dev_backlog, sw_backlog = ljm.eStreamRead(self.handle)
            chunk = np.array(chunk).reshape((scansPerRead, numAddresses)).T
            self._databuffer[:, i * scansPerRead : (i + 1) * scansPerRead] = chunk
            if self._stop_buffer:
                break
            if sw_backlog < scansPerRead:
                await asyncio.sleep(0.1)

        print("stream_worker finished", file=self.log_debug)
        ljm.eStreamStop(self.handle)
        self.set_state(DevState.ON)

        # PROTECTED REGION END #    //  LabJackT7.init_device

    # def always_executed_hook(self):
    #     """Method always executed before any TANGO command is executed."""
    # PROTECTED REGION ID(LabJackT7.always_executed_hook) ENABLED START #
    # PROTECTED REGION END #    //  LabJackT7.always_executed_hook

    def delete_device(self):
        """Hook to delete resources allocated in init_device.

        This method allows for any memory or other resources allocated in the
        init_device method to be released.  This method is called by the device
        destructor and by the device Init command.
        """
        ljm.close(self.handle)
        # PROTECTED REGION ID(LabJackT7.delete_device) ENABLED START #
        # PROTECTED REGION END #    //  LabJackT7.delete_device

    # ------------------
    # Attributes methods
    # ------------------

    def read_AIN0(self):
        # PROTECTED REGION ID(LabJackT7.AIN0_read) ENABLED START #
        """Return the AIN0 attribute."""
        return self.read_AIN(0)
        # PROTECTED REGION END #    //  LabJackT7.AIN0_read

    def read_AIN1(self):
        # PROTECTED REGION ID(LabJackT7.AIN1_read) ENABLED START #
        """Return the AIN1 attribute."""
        return self.read_AIN(1)
        # PROTECTED REGION END #    //  LabJackT7.AIN1_read

    def read_AIN2(self):
        # PROTECTED REGION ID(LabJackT7.AIN2_read) ENABLED START #
        """Return the AIN2 attribute."""
        return self.read_AIN(2)
        # PROTECTED REGION END #    //  LabJackT7.AIN2_read

    def read_AIN3(self):
        # PROTECTED REGION ID(LabJackT7.AIN3_read) ENABLED START #
        """Return the AIN3 attribute."""
        return self.read_AIN(3)
        # PROTECTED REGION END #    //  LabJackT7.AIN3_read

    def read_buffer_scanrate(self):
        # PROTECTED REGION ID(LabJackT7.buffer_scanrate_read) ENABLED START #
        """Return the buffer_scanrate attribute."""
        return self._buffer_scanrate
        # PROTECTED REGION END #    //  LabJackT7.buffer_scanrate_read

    def write_buffer_scanrate(self, value):
        # PROTECTED REGION ID(LabJackT7.buffer_scanrate_write) ENABLED START #
        """Set the buffer_scanrate attribute."""
        self._buffer_scanrate = value
        # PROTECTED REGION END #    //  LabJackT7.buffer_scanrate_write

    def read_buffer_size(self):
        # PROTECTED REGION ID(LabJackT7.buffer_size_read) ENABLED START #
        """Return the buffer_size attribute."""
        return self._buffer_size
        # PROTECTED REGION END #    //  LabJackT7.buffer_size_read

    def write_buffer_size(self, value):
        # PROTECTED REGION ID(LabJackT7.buffer_size_write) ENABLED START #
        """Set the buffer_size attribute."""
        self._buffer_size = value
        # PROTECTED REGION END #    //  LabJackT7.buffer_size_write

    def is_buffer_size_allowed(self, attr):
        # PROTECTED REGION ID(LabJackT7.is_buffer_size_allowed) ENABLED START #
        if attr == attr.READ_REQ:
            return True
        else:
            return self.get_state() not in [DevState.RUNNING]
        # PROTECTED REGION END #    //  LabJackT7.is_buffer_size_allowed

    def read_arrayAIN0(self):
        # PROTECTED REGION ID(LabJackT7.arrayAIN0_read) ENABLED START #
        """Return the arrayAIN0 attribute."""
        return self._databuffer[0, : self._buffer_size]
        # PROTECTED REGION END #    //  LabJackT7.arrayAIN0_read

    def read_arrayAIN1(self):
        # PROTECTED REGION ID(LabJackT7.arrayAIN1_read) ENABLED START #
        """Return the arrayAIN1 attribute."""
        return self._databuffer[1, : self._buffer_size]
        # PROTECTED REGION END #    //  LabJackT7.arrayAIN1_read

    def read_arrayAIN2(self):
        # PROTECTED REGION ID(LabJackT7.arrayAIN2_read) ENABLED START #
        """Return the arrayAIN2 attribute."""
        return self._databuffer[2, : self._buffer_size]
        # PROTECTED REGION END #    //  LabJackT7.arrayAIN2_read

    def read_arrayAIN3(self):
        # PROTECTED REGION ID(LabJackT7.arrayAIN3_read) ENABLED START #
        """Return the arrayAIN3 attribute."""
        return self._databuffer[3, : self._buffer_size]
        # PROTECTED REGION END #    //  LabJackT7.arrayAIN3_read

    # --------
    # Commands
    # --------

    @command()
    @DebugIt()
    async def readAINbuffered(self):
        # PROTECTED REGION ID(LabJackT7.readAINbuffered) ENABLED START #
        """
        Start a buffered, fixed-rate measurement on given channels.

        :return:None
        """
        params = {
            "STREAM_TRIGGER_INDEX": 0,  # Ensure triggered stream is disabled.
            "STREAM_CLOCK_SOURCE": 0,  # Enabling internally-clocked stream.
            "STREAM_SETTLING_US": 0,
            "STREAM_RESOLUTION_INDEX": 0,
            # "AIN_ALL_NEGATIVE_CH": ljm.constants.GND,
        }
        self.set_params_from_dict(params)
        self._stop_buffer = False
        asyncio.create_task(self.stream_worker())
        # PROTECTED REGION END #    //  LabJackT7.readAINbuffered

    def is_readAINbuffered_allowed(self):
        # PROTECTED REGION ID(LabJackT7.is_readAINbuffered_allowed) ENABLED START #
        return self.get_state() not in [DevState.RUNNING]
        # PROTECTED REGION END #    //  LabJackT7.is_readAINbuffered_allowed

    @command()
    def stop_buffer(self):
        self._stop_buffer = True

    @command(
        dtype_in="DevVarLongArray",
        doc_in="channels",
        dtype_out="DevVarFloatArray",
        doc_out="voltages",
    )
    @DebugIt()
    def readAINmulti(self, argin):
        # PROTECTED REGION ID(LabJackT7.readAINmulti) ENABLED START #
        """
        Read analog voltages from a list of channels

        :param argin: 'DevVarLongArray'
        channels

        :return:'DevVarFloatArray'
        voltages
        """
        channelnames = [f"AIN{ch}" for ch in argin]
        values = ljm.eReadNames(self.handle, len(argin), channelnames)
        return np.array(values)
        # PROTECTED REGION END #    //  LabJackT7.readAINmulti

    def is_readAINmulti_allowed(self):
        # PROTECTED REGION ID(LabJackT7.is_readAINmulti_allowed) ENABLED START #
        return self.get_state() not in [DevState.RUNNING]
        # PROTECTED REGION END #    //  LabJackT7.is_readAINmulti_allowed

    @command(
        dtype_in="DevVarLongArray",
        doc_in="channel_config",
    )
    @DebugIt()
    def write_channel_config(self, argin):
        # PROTECTED REGION ID(LabJackT7.write_channel_config) ENABLED START #
        """
            channel configuration:
            [
              channel number (1-4),
              negative channel number (1-13, 199 for GND),
              measurement range (0=``10 V``, 1=``1 V``, 2=``0.1 V``, 3=``0.01 V``),
              resolution index (0-8),
              settling time (µs),
            ]

        :param argin: 'DevVarLongArray'
        channel_config

        :return:None
        """
        channel, negative, measrange, resolution = argin
        config = {
            f"AIN{channel}_NEGATIVE_CH": negative,
            f"AIN{channel}_RANGE": 10 ** (1 - measrange),
            f"AIN{channel}_RESOLUTION_INDEX": resolution,
            f"AIN{channel}_SETTLING_US": 0,
        }
        self.set_params_from_dict(config)
        # PROTECTED REGION END #    //  LabJackT7.write_channel_config

    def is_write_channel_config_allowed(self):
        # PROTECTED REGION ID(LabJackT7.is_write_channel_config_allowed) ENABLED START #
        return self.get_state() not in [DevState.RUNNING]
        # PROTECTED REGION END #    //  LabJackT7.is_write_channel_config_allowed

    @command(
        dtype_in="DevLong",
        doc_in="channel",
        dtype_out="DevVarLongArray",
        doc_out="channel_config",
    )
    @DebugIt()
    def read_channel_config(self, argin):
        # PROTECTED REGION ID(LabJackT7.read_channel_config) ENABLED START #
        """
        Returns channel configuration. See write_channel_config for format.

        :param argin: 'DevLong'
        channel

        :return:'DevVarLongArray'
        channel_config
        """
        return [0]
        # PROTECTED REGION END #    //  LabJackT7.read_channel_config

    def is_read_channel_config_allowed(self):
        # PROTECTED REGION ID(LabJackT7.is_read_channel_config_allowed) ENABLED START #
        return self.get_state() not in [DevState.RUNNING]
        # PROTECTED REGION END #    //  LabJackT7.is_read_channel_config_allowed

    @command()
    async def connect(self):
        print(f"{self.connection_type=}, {self.IPaddress=}", file=self.log_debug)
        self.handle = ljm.openS("ANY", self.connection_type, self.IPaddress)
        print(f"Acquired handle {self.handle}", file=self.log_debug)
        self.set_state(DevState.ON)

    def is_connect_allowed(self):
        # PROTECTED REGION ID(LabJackT7.is_connect_allowed) ENABLED START #
        return self.get_state() not in [DevState.RUNNING]
        # PROTECTED REGION END #    //  LabJackT7.is_connect_allowed


# ----------
# Run server
# ----------


def main(args=None, **kwargs):
    """Main function of the LabJackT7 module."""
    # PROTECTED REGION ID(LabJackT7.main) ENABLED START #
    return run((LabJackT7,), args=args, **kwargs)
    # PROTECTED REGION END #    //  LabJackT7.main


if __name__ == "__main__":
    main()
