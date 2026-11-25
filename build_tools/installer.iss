; PLEAP v1.12 Setup script - Inno Setup 6.3+ compatible

#define MyAppName       "PLEAP"
#define MyAppFull       "Productivity Lab Efficiency Analysis Program"
#define MyAppVersion    "1.12"
#define MyAppPublisher  "Incheon National University, Productivity Lab"
#define MyAppEmail      "ckim@inu.ac.kr"
#define MyAppExeName    "PLEAP.exe"

[Setup]
AppId={{B7E5C0D2-9F3E-4B1A-8C7B-PLEAP00104}}
AppName={#MyAppName}
AppVerName={#MyAppName} {#MyAppVersion}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL=mailto:{#MyAppEmail}
AppSupportURL=mailto:{#MyAppEmail}
AppContact={#MyAppEmail}
AppCopyright=Copyright (C) Incheon National University, Productivity Lab
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppFull}
VersionInfoVersion={#MyAppVersion}.0
VersionInfoCopyright=Copyright (C) Incheon National University, Productivity Lab

DefaultDirName={autopf}\PLEAP
DefaultGroupName=PLEAP
DisableProgramGroupPage=yes
OutputDir=..\Output
OutputBaseFilename=PLEAP_Setup_v{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
WizardStyle=modern
SetupIconFile=..\src\pleap\assets\lab_logo.ico
UninstallDisplayIcon={app}\lab_logo.ico
UninstallDisplayName={#MyAppName} {#MyAppVersion}
LicenseFile=..\LICENSE.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "korean";  MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\PLEAP\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
Source: "..\src\pleap\assets\lab_logo.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\examples\*"; DestDir: "{app}\examples"; Flags: recursesubdirs ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\lab_logo.ico"; WorkingDir: "{app}"
Name: "{group}\License"; Filename: "{app}\LICENSE.txt"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\lab_logo.ico"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
