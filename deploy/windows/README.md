# Windows ECS Deployment Guide

如果你现在的阿里云 ECS 是 `Windows Server`，这一版按更正式的上线流程来：

- 远程信号服务平台
- IIS 托管前端静态页面
- 后端通过计划任务开机自启
- 你自己在券商 APP 手动下单

## 推荐目录

- 代码目录：`D:\ashare-quant\app`
- 数据库目录：`D:\ashare-quant\db`
- 运行数据：`D:\ashare-quant\runtime`
- 日志目录：`D:\ashare-quant\logs`
- 备份目录：`D:\ashare-quant\backups`
- 前端发布目录：`D:\ashare-quant\sites\frontend`
- 如果你已经在 IIS 中使用其他站点目录，例如 `D:\wwwroot\lianghua-web`，后面的脚本也支持直接指定

建议：

- `C:` 放系统、Python、Node、Git
- `D:` 放代码、数据库、日志、备份

## 1. 安装依赖

在 Windows ECS 上安装：

- Git
- Python 3.12
- Node.js LTS

安装完成后，在 PowerShell 中确认：

```powershell
python --version
node --version
npm --version
git --version
```

## 2. 准备目录

管理员 PowerShell：

```powershell
New-Item -ItemType Directory -Force -Path D:\ashare-quant\app
New-Item -ItemType Directory -Force -Path D:\ashare-quant\db
New-Item -ItemType Directory -Force -Path D:\ashare-quant\runtime
New-Item -ItemType Directory -Force -Path D:\ashare-quant\logs
New-Item -ItemType Directory -Force -Path D:\ashare-quant\backups
```

## 3. 上传项目

把本项目放到：

`D:\ashare-quant\app`

最终结构应当类似：

- `D:\ashare-quant\app\backend`
- `D:\ashare-quant\app\frontend`

## 4. 配置后端 `.env`

编辑：

`D:\ashare-quant\app\backend\.env`

推荐起步配置：

```env
DATA_PROVIDER=tushare
TUSHARE_TOKEN=你的token
FRONTEND_ORIGIN=http://116.62.21.194:8080
FRONTEND_ORIGINS=http://116.62.21.194:8080
DATABASE_URL=D:/ashare-quant/db/ashare_quant.db
DATA_DIR=D:/ashare-quant/runtime
UNIVERSE_LIMIT=50
UNIVERSE_SOURCE=index
UNIVERSE_INDEX_CODE=000300
UNIVERSE_INDEX_NAME=沪深300
FALLBACK_TO_MOCK_ON_DATA_ERROR=true
```

说明：

- 线上 Windows ECS 使用 `backend/.env`
- 本地开发使用 `backend/.env.local`
- 后端已经按 `.env -> .env.local` 顺序加载，本地文件会覆盖线上默认值
- 不要把 `backend/.env.local` 提交到远程仓库，也不要放到 ECS 上

## 5. 初始化后端

```powershell
cd D:\ashare-quant\app
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\prepare_backend.ps1
```

## 6. 配置后端开机自启

```powershell
cd D:\ashare-quant\app\backend
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\install_backend_task.ps1
```

检查：

- 本机：`http://127.0.0.1:8000/health`
- 外部：`http://116.62.21.194:8000/health`

如果外部打不开，放行阿里云安全组 `8000` 端口。

后端日志默认写到：

- `D:\ashare-quant\logs\backend.out.log`
- `D:\ashare-quant\logs\backend.err.log`

## 7. 启用 IIS

管理员 PowerShell：

```powershell
Install-WindowsFeature Web-Server,Web-Http-Redirect,Web-Static-Content,Web-Default-Doc,Web-Http-Errors,Web-Http-Logging,Web-Filtering,Web-Performance,Web-Stat-Compression,Web-Dyn-Compression,Web-Mgmt-Tools,Web-Scripting-Tools
```

如果你要让 Vue 单页路由正常工作，还需要在 IIS 上安装 `URL Rewrite` 组件。

## 8. 发布前端到 IIS

```powershell
cd D:\ashare-quant\app\backend
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\build_frontend.ps1 -PublishRoot "D:\wwwroot\lianghua-web"
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\install_iis_site.ps1 -SitePath "D:\wwwroot\lianghua-web" -Port 8080
```

检查：

- 本机：`http://127.0.0.1:8080/`
- 外部：`http://116.62.21.194:8080/`

如果外部打不开，放行阿里云安全组 `8080` 端口。

说明：

- 线上前端构建读取 `frontend/.env.production`
- 本地前端开发读取 `frontend/.env.local`
- 不要把 `frontend/.env.local` 提交到远程仓库

## 9. 配置每日自动 sync/train

推荐在交易日收盘后自动执行，例如每天 `18:00`：

```powershell
cd D:\ashare-quant\app\backend
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\install_refresh_task.ps1 -DailyTime "18:00"
```

刷新日志默认写到：

- `D:\ashare-quant\logs\refresh.out.log`
- `D:\ashare-quant\logs\refresh.err.log`

## 10. 上线后你应该看到什么

前端入口：

- `http://116.62.21.194:8080/overview`
- `http://116.62.21.194:8080/signals`

后端接口：

- `http://116.62.21.194:8000/health`

## 11. 日常更新命令

后端更新后：

```powershell
cd D:\ashare-quant\app\backend
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\update_backend.ps1
```

如果这次改了 Python 依赖，带上：

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\update_backend.ps1 -InstallDeps
```

如果这次改了数据/模型逻辑，带上：

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\update_backend.ps1 -RunRefresh
```

前端更新后：

```powershell
cd D:\ashare-quant\app\backend
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\update_frontend.ps1 -PublishRoot "D:\wwwroot\lianghua-web"
```

如果这次改了前端依赖，带上：

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\update_frontend.ps1 -PublishRoot "D:\wwwroot\lianghua-web" -InstallDeps
```

只想重启后端服务时：

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\restart_backend.ps1
```

## 12. 当前需要修改、添加、删除的文件

这次按标准上线流程，建议这样整理：

需要保留并使用：

- `backend/.env`
- `frontend/.env.production`
- `deploy/windows/scripts/prepare_backend.ps1`
- `deploy/windows/scripts/start_backend.ps1`
- `deploy/windows/scripts/restart_backend.ps1`
- `deploy/windows/scripts/install_backend_task.ps1`
- `deploy/windows/scripts/update_backend.ps1`
- `deploy/windows/scripts/update_frontend.ps1`
- `deploy/windows/scripts/run_refresh.ps1`
- `deploy/windows/scripts/install_refresh_task.ps1`
- `deploy/windows/scripts/build_frontend.ps1`
- `deploy/windows/scripts/install_iis_site.ps1`
- `deploy/windows/iis/web.config`

需要替换旧使用方式：

- 不再用 `vite preview` 作为正式前端服务
- 不再让后端启动脚本每次都重新安装依赖

## 13. 后续再补的生产项

后面可以继续加：

- SQLite 定时备份任务
- IIS HTTPS
- Windows 防火墙规则
- 后端进程健康监控
