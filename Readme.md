# LabJackT7

Tango device server to read analog inputs on LabJack T7 via ethernet


## Requirements

* pip: labjack-ljm
* ljm library from [labjack homepage](https://labjack.com/pages/support?doc=/software-driver/installer-downloads/ljm-software-installers-t4-t7-digit/)

## Configuration

Currently, only communication via ethernet is supported. The device also offers direct USB communication, but at reduced data rates and, in our testing, less stability.

* device property `IPaddress`

## Implemented features

* default configuration of analog input ports
* custom channel config via `write_channel_config` command
* single measurements on AIN1-4 via attributes
* buffered, fixed-rate measurements on AIN1-4
  * configure buffer size and acquisition rate via attributes
  * start/ abort measurement via commands


## Device documentation

* [ljm library](https://labjack.com/pages/support/?doc=/software-driver/ljm-users-guide/)
* [LabJack on github](https://github.com/labjack)
