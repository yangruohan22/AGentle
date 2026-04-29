import importlib
import importlib.metadata
import warnings

warnings.filterwarnings("ignore")

print("🔍 正在扫描当前环境中的【第三方依赖包】，开始精准排雷...\n")

broken_modules = []
success_count = 0

for dist in importlib.metadata.distributions():
    try:
        # 获取包名，如果包名读取失败，直接跳过
        package_name = dist.metadata.get('Name')
        if package_name is None:
            continue

        # 尝试读取导入名
        top_levels = dist.read_text('top_level.txt')
        if top_levels:
            import_names = top_levels.split()
        else:
            import_names = [package_name.replace('-', '_')]

        for mod_name in import_names:
            if mod_name.startswith('_'):
                continue

            try:
                importlib.import_module(mod_name)
                success_count += 1
                print(f"\r✅ 正在检测... {success_count} 个模块健康通过", end="", flush=True)
            except (ImportError, ModuleNotFoundError) as e:
                # 核心：只记录那些确实存在但在底层报错的包
                broken_modules.append((package_name, mod_name, str(e)))
            except Exception:
                pass
    except Exception:
        # 遇到任何读取 Metadata 的错误直接跳过，说明该包已经损坏到不可救药，需手动处理
        continue

print(f"\n\n🏁 扫描结束！成功加载了 {success_count} 个健康的模块。")
print("-" * 60)

if not broken_modules:
    # 如果没扫出来，但你跑主程序还是报错，说明我们需要手动强制重装几个核心库
    print("🎉 扫描器未发现明显的底层损坏。")
    print("💡 建议：由于环境曾中断，请手动运行以下命令保底修复核心科学计算库：")
    print(
        "pip install --force-reinstall numpy scipy PyWavelets neurokit2 mne scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple")
else:
    print(f"🚨 警报！发现了 {len(broken_modules)} 个损坏的包：\n")
    damaged_packages = set()
    for pkg, mod, err in broken_modules:
        print(f"❌ [包名]: {pkg:<15} | [模块]: {mod:<15} \n   [报错]: {err}")
        damaged_packages.add(pkg)

    print("-" * 60)
    print("💡 【一键修复命令】请复制并在终端运行以下命令：")
    fix_cmd = "pip install --force-reinstall " + " ".join(
        damaged_packages) + " -i https://pypi.tuna.tsinghua.edu.cn/simple"
    print(fix_cmd)