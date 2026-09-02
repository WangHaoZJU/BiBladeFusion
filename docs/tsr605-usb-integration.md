# TSR605 USB integration boundary

## Current decision

Thermal acquisition remains disabled and fail-closed. The SDK copied to
`/home/vale/Documents/Proj1/HCNetSDK` is **Hikvision Device Network SDK**, not a reviewed
direct-USB binding that can enumerate and open a locally attached TSR605 radiometric
module. It must not be loaded as though it had established that direct-USB contract.

This does not mean HCNetSDK has no thermal-camera features. It means the thermal
features found in this package are reached through a logged-in network device and do not
establish a supported local-USB path for this camera. A hypothetical device mode that
presents a network interface over USB (for example USB-RNDIS) has not been ruled out, but
the supplied material does not identify TSR605 support, the required address/login mode,
or a validated radiometric payload for such a setup. It therefore remains blocked too.

## Local evidence reviewed on 2026-09-01

The following evidence comes from the supplied package itself; no native library was
loaded during the audit.

- `开发文档/设备网络SDK使用手册.chm` identifies the package family.
- `头文件/HCNetSDK.h` declares `NET_DVR_Login_V40`. Its login structure contains a
  device address, TCP port, username and password.
- The same header declares `NET_DVR_RealPlay_V40`, whose preview structure is described
  in terms of channels and network streaming protocols.
- `NET_DVR_CaptureJPEGPicture_WithAppendData` can return a JPEG plus a full-screen
  thermometry payload, but it accepts the `lUserID` produced by network login and a
  network-device channel. This is not local USB enumeration or open-by-serial.
- Header mentions of USB include USB ports, storage devices, USB cameras managed by other
  Hikvision products, and alarm/log fields. They are not evidence of a host-side TSR605
  radiometric capture API.
- No file or example names TSR605. No reviewed API in the package enumerates a local USB
  thermal module, pins its serial number, or defines the byte layout and conversion of a
  TSR605 per-pixel Celsius frame.

Audited file identities:

```text
HCNetSDK.h      sha256 50de3eeb87ef4311ab23ffe84de3f721318b847d704f5e8744fde60cc00f2484
libhcnetsdk.so  sha256 c710cf6c5cad8cad0117de4e0c056471bf331666dbab241cf381f2d83000f572
```

A reviewed direct-USB Linux64 SDK package, a radiometric USB example, and an explicit
TSR605 compatibility statement are not present in the supplied directory. Their exact
vendor product name and API must come from the camera supplier rather than being inferred
from HCNetSDK filenames or unrelated USB fields.

## Implemented project boundary

The project now has the following pieces without claiming a working hardware driver:

- `bbf thermal audit-sdk` classifies an SDK directory without loading shared libraries or
  touching a USB device. The supplied HCNetSDK is reported as incompatible and the
  command exits nonzero.
- `thermal.driver: tsr605_usb` is the only recognized future driver name. Model,
  transport, serial number, timeout, and optional expected radiometric dimensions have
  explicit configuration fields.
- `Tsr605UsbBackend` is a narrow internal seam for a future implementation based only on
  the official USB SDK. The wrapper pins model and serial, requires a per-pixel Celsius
  matrix, validates the configured dimensions, and fails closed on missing/invalid data.
- Synchronized session storage preserves the temperature matrix, optional raw counts, and
  camera/SDK identity provenance.
- `bbf acquire snapshot` composes the configured thermal camera. If thermal acquisition
  is enabled before a reviewed USB backend exists, it exits before opening the robot or
  creating a capture session.
- The unknown-blade motion runtime still requires thermal to be disabled. Adding this
  adapter boundary does not authorize thermal fusion or robot motion with a new payload.

Run the current local audit from the repository root:

```bash
uv run bbf thermal audit-sdk \
  --sdk-root /home/vale/Documents/Proj1/HCNetSDK \
  --json
```

Expected result: `kind` is `hikvision_device_network_sdk`, `compatible` is `false`, and
the process exits with status 1.

## Required vendor package and true-hardware inputs

Do not set `thermal.enabled: true` until all of these are available and recorded:

1. Official TSR605-compatible **USB SDK Linux64**, including headers, native libraries,
   and the vendor's radiometric USB example.
2. Vendor documentation for local USB enumeration, open-by-serial, capture/trigger,
   timeout semantics, frame timestamp, per-pixel data layout, temperature unit/scale,
   invalid-pixel encoding, and ownership/lifetime of callback buffers.
3. The physical TSR605 serial number and verified radiometric width/height.
4. USB access setup for the runtime user (VID/PID and a least-privilege udev rule, if the
   official SDK requires one). Do not run the camera stack as root merely to bypass USB
   permissions.
5. The camera's supported measurement range and the physical measurement settings needed
   for accuracy: emissivity, reflected apparent temperature, target distance,
   atmospheric temperature/humidity, optics and calibration identity.
6. A thermal intrinsic calibration and a validated rigid transform from the thermal
   optical frame to the robot payload/geometry frame.
7. A replacement ES68 payload collision model that includes the D435i, TSR605, mount and
   cables, followed by renewed collision and motion-envelope acceptance.
8. An independently measured radiometric acceptance dataset before temperatures can be
   treated as scientific results.

After the correct package arrives, first add a vendor-specific backend behind
`Tsr605UsbBackend` and hardware-marked tests. Only then may the SDK audit recognize that
exact reviewed package. Do not parse palette-coloured UVC video as temperature, do not
guess a raw-count-to-Celsius formula, and do not use HCNetSDK network thermometry structs
as a USB ABI.

## Scope that remains deliberately blocked

The following are not implemented or authorized by this work:

- thermal-to-D435i/robot hand-eye calibration;
- geometric projection of thermal pixels onto the blade surface;
- radiometric correction or uncertainty modelling;
- thermal TSDF or thermal texture fusion;
- unknown-blade scan execution with the thermal camera mounted;
- production/scientific acceptance of any TSR605 temperature data.
